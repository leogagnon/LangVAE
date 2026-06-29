"""
Train a LangVAE sentence VAE on FineWeb.

Usage:
    # Local single-GPU
    python train.py

    # Local multi-GPU
    torchrun --nproc_per_node=4 train.py

    # Single SLURM job — launcher auto-detected from CC_CLUSTER env var
    python train.py

    # Single SLURM job — explicit launcher
    python train.py hydra/launcher=fir

    # Hyperparameter sweep on SLURM
    python train.py --multirun latent_size=128,256 hydra/launcher=fir
"""
import os
import random
import subprocess
import sys
import time

import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from pythae.data.datasets import DatasetOutput
from pythae.models.vae import VAEConfig
from torch.utils.data import Dataset

from langvae import LangVAE
from langvae.encoders import SentenceEncoder
from langvae.decoders import SentenceDecoder
from langvae.pipelines import LanguageTrainingPipeline
from langvae.trainers import CyclicalScheduleKLThresholdTrainerConfig
from langvae.trainers.training_callbacks import TensorBoardCallback

torch.set_float32_matmul_precision("high")


class FineWebSpanDataset(Dataset):
    """Map-style dataset over FineWeb that samples random spans online.

    Each call to __getitem__ draws a random document from FineWeb, selects a
    random contiguous window of char_window characters within it, tokenizes that
    window, then slices a random span of min_len–max_len tokens.
    Returns a DatasetOutput with a sparse one-hot input_ids tensor, matching the
    format expected by LangVAE's collate_sparse_fn and trainer.

    __len__ returns `size`, which controls how many steps constitute one epoch.
    Documents are sampled with replacement so the effective dataset is infinite.
    """

    def __init__(self, hf_dataset, tokenizer, size: int, min_len: int, max_len: int):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.size = size
        self.min_len = min_len
        self.max_len = max_len
        self.char_window = max_len * 12
        self.vocab_size = len(tokenizer.get_vocab())
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, _idx) -> DatasetOutput:
        doc_idx = random.randrange(len(self.dataset))
        text = self.dataset[doc_idx]["text"]

        if len(text) > self.char_window:
            char_start = random.randint(0, len(text) - self.char_window)
            text = text[char_start : char_start + self.char_window]

        ids = self.tokenizer(text, truncation=False, padding=False)["input_ids"]
        n = len(ids)

        span_len = random.randint(min(self.min_len, n), min(n, self.max_len))
        start = random.randint(0, n - span_len)
        ids = ids[start : start + span_len]

        input_ids = torch.sparse_coo_tensor(
            [list(range(len(ids))), ids],
            [1] * len(ids),
            (len(ids), self.vocab_size),
            dtype=torch.int8,
        )
        attention_mask = torch.ones(len(ids), dtype=torch.long)
        return DatasetOutput(data=input_ids, input_ids=input_ids, attention_mask=attention_mask)


def _copy_dataset_to_tmpdir() -> None:
    """Copy FineWeb from its cache location to SLURM_TMPDIR.

    Reads FINEWEB_CACHE_DIR for the source (falls back to $HF_HOME/datasets).
    After the copy, sets FINEWEB_CACHE_DIR to the tmpdir so load_dataset
    picks it up automatically.
    Whichever process creates the sentinel first does the copy; all others wait.
    Must be called before any HuggingFace dataset load.
    """
    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    if not slurm_tmpdir:
        raise RuntimeError("copy_dataset_to_tmpdir=true but SLURM_TMPDIR is not set")

    src_cache_dir = os.environ.get("FINEWEB_CACHE_DIR")
    if src_cache_dir is None:
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        src_cache_dir = os.path.join(hf_home, "datasets")

    sentinel = os.path.join(slurm_tmpdir, ".fineweb_copy_done")

    try:
        fd = os.open(sentinel, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        is_copier = True
    except FileExistsError:
        is_copier = False

    if is_copier:
        find = subprocess.run(
            ["find", "HuggingFaceFW___fineweb/", "-type", "f"],
            cwd=src_cache_dir, capture_output=True, text=True, check=True,
        )
        n_files = find.stdout.count("\n")
        print(f"Copier: copying {n_files} FineWeb files to {slurm_tmpdir} ...", flush=True)
        t0 = time.time()
        subprocess.run(
            ["parallel", "-j", "8", "cp", "-n", "--parents", "{}", slurm_tmpdir],
            input=find.stdout, text=True, cwd=src_cache_dir, stderr=subprocess.DEVNULL,
        )
        with open(sentinel, "w") as f:
            f.write("done")
        print(f"Copier: done in {time.time() - t0:.1f}s", flush=True)
    else:
        while os.path.getsize(sentinel) == 0:
            time.sleep(1)

    os.environ["FINEWEB_CACHE_DIR"] = slurm_tmpdir


def _load_fineweb() -> object:
    cache_dir = os.environ.get("FINEWEB_CACHE_DIR")
    print("Loading FineWeb from local cache...", flush=True)
    t0 = time.monotonic()
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-100BT",
        split="train",
        streaming=False,
        cache_dir=cache_dir,
    )
    print(f"load_dataset took {time.monotonic() - t0:.1f}s", flush=True)
    return dataset


def _main(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    exp_label = (
        f"fineweb-langvae"
        f"-{cfg.encoder_model.replace('/', '--')}"
        f"-{cfg.decoder_model.replace('/', '--')}"
        f"-l{cfg.latent_size}"
    )

    hf_dataset = _load_fineweb()

    print("Loading decoder...", flush=True)
    decoder = SentenceDecoder(
        cfg.decoder_model, cfg.latent_size, cfg.max_sent_len,
        device=device, device_map="auto",
    )
    print("Loading encoder...", flush=True)
    encoder = SentenceEncoder(
        cfg.encoder_model, cfg.latent_size, decoder.tokenizer, device=device,
    )

    train_dataset = FineWebSpanDataset(
        hf_dataset, decoder.tokenizer,
        size=cfg.num_train, min_len=cfg.min_sent_len, max_len=cfg.max_sent_len,
    )
    eval_dataset = FineWebSpanDataset(
        hf_dataset, decoder.tokenizer,
        size=cfg.num_eval, min_len=cfg.min_sent_len, max_len=cfg.max_sent_len,
    )

    model = LangVAE(VAEConfig(latent_dim=cfg.latent_size), encoder, decoder)

    training_config = CyclicalScheduleKLThresholdTrainerConfig(
        output_dir=exp_label,
        num_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        train_dataloader_num_workers=cfg.num_dataloader_workers,
        eval_dataloader_num_workers=cfg.num_dataloader_workers,
        steps_saving=cfg.steps_saving,
        optimizer_cls=cfg.optimizer,
        scheduler_cls=cfg.scheduler,
        scheduler_params={"patience": cfg.scheduler_patience, "factor": cfg.scheduler_factor},
        start_beta=cfg.start_beta,
        max_beta=cfg.max_beta,
        n_cycles=cfg.n_cycles,
        target_kl=cfg.target_kl,
    )

    pipeline = LanguageTrainingPipeline(training_config=training_config, model=model)
    tb_callback = TensorBoardCallback(exp_label)
    pipeline(train_data=train_dataset, eval_data=eval_dataset, callbacks=[tb_callback])


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def _hydra_main(cfg: DictConfig) -> None:
    # When submitit launches one SLURM task per GPU, map SLURM rank vars to
    # the env vars torch.distributed expects so DDP initialises correctly.
    in_slurm = "SLURM_JOB_ID" in os.environ
    already_distributed = "LOCAL_RANK" in os.environ

    if in_slurm and not already_distributed and int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        master_port = str(29500 + int(os.environ.get("SLURM_JOB_ID", 0)) % 10000)
        os.environ.update({
            "LOCAL_RANK":  os.environ["SLURM_LOCALID"],
            "RANK":        os.environ["SLURM_PROCID"],
            "WORLD_SIZE":  os.environ["SLURM_NTASKS"],
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": master_port,
        })

    # Disable tokenizers' internal thread pool — deadlocks with DataLoader workers.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if cfg.get("copy_dataset_to_tmpdir", False):
        _copy_dataset_to_tmpdir()

    _main(cfg)


_CC_LAUNCHER_MAP = {"fir": "fir", "nibi": "nibi", "tamia": "tamia"}

if __name__ == "__main__":
    cc_cluster = os.environ.get("CC_CLUSTER")
    if cc_cluster in _CC_LAUNCHER_MAP and not any(
        "hydra/launcher" in arg for arg in sys.argv[1:]
    ):
        sys.argv.append(f"hydra/launcher={_CC_LAUNCHER_MAP[cc_cluster]}")
    _hydra_main()
