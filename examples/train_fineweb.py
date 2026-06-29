"""
Train a Flan-T5-base + LLaMA-3.2-3B VAE on FineWeb for use as the sentence encoder in STAR-LDM.

LATENT_SIZE=768 matches Flan-T5-base's hidden size and STAR-LDM's existing score net /
soft prompt generator dimensions, so those components need no architectural changes.

Launch:
    Single GPU:
        python examples/train_fineweb.py

    4x H100:
        torchrun --nproc_per_node=4 examples/train_fineweb.py

    SLURM (copy FineWeb to tmpdir first):
        COPY_TO_TMPDIR=1 torchrun --nproc_per_node=4 examples/train_fineweb.py

Environment variables:
    FINEWEB_CACHE_DIR   Override the HF datasets cache location for FineWeb only.
                        Mirrors the same variable used by STAR-LDM's get_fineweb_dataset.
    COPY_TO_TMPDIR      Set to 1 to copy FineWeb to SLURM_TMPDIR before loading
                        (rank 0 copies, others wait on sentinel file).
"""
import os
import random
import subprocess
import time

import torch
from datasets import load_dataset
from pythae.models.vae import VAEConfig
from pythae.data.datasets import DatasetOutput
from torch import Tensor
from torch.utils.data import Dataset

from langvae import LangVAE
from langvae.encoders import SentenceEncoder
from langvae.decoders import SentenceDecoder
from langvae.pipelines import LanguageTrainingPipeline
from langvae.trainers import CyclicalScheduleKLThresholdTrainerConfig
from langvae.trainers.training_callbacks import TensorBoardCallback

ENCODER_MODEL = "google/flan-t5-base"
DECODER_MODEL = "meta-llama/Llama-3.2-3B"
LATENT_SIZE = 768   # matches Flan-T5-base hidden size and STAR-LDM's existing architecture
MAX_SENT_LEN = 128  # matches STAR-LDM's MAX_LENGTH
MIN_SENT_LEN = 16
CHAR_WINDOW = MAX_SENT_LEN * 12  # character window fed to the tokenizer; generous enough
                                  # to always yield >= MAX_SENT_LEN tokens for any long doc
NUM_TRAIN = 5_000_000
NUM_EVAL = 50_000
NUM_EPOCHS = 5
BATCH_SIZE = 64

# target_kl scales with latent_size relative to LangVAE's default 128-dim / 2.0 setting
TARGET_KL = 2.0 * LATENT_SIZE / 128   # = 12.0
N_CYCLES = 40

NUM_DATALOADER_WORKERS = 8  # one per CPU core available for data loading; tune to your node

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXP_LABEL = f"fineweb-langvae-flan-t5-base-llama3.2-3b-l{LATENT_SIZE}"

torch.set_float32_matmul_precision("high")
# Disable the tokenizers library's internal thread pool: it deadlocks when combined
# with DataLoader's fork-based worker processes.
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class FineWebSpanDataset(Dataset):
    """Map-style dataset over FineWeb that samples random spans online.

    Each call to __getitem__ draws a random document from FineWeb, selects a
    random contiguous window of CHAR_WINDOW characters within it, tokenizes that
    window, then slices a random span of MIN_SENT_LEN–MAX_SENT_LEN tokens.
    Returns a DatasetOutput with a sparse one-hot input_ids tensor, matching the
    format expected by LangVAE's collate_sparse_fn and trainer.

    __len__ returns `size`, which controls how many steps constitute one epoch.
    Documents are sampled with replacement so the effective dataset is infinite.
    """

    def __init__(self, hf_dataset, tokenizer, size: int):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.size = size
        self.vocab_size = len(tokenizer.get_vocab())
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, _idx) -> DatasetOutput:
        doc_idx = random.randrange(len(self.dataset))
        text = self.dataset[doc_idx]["text"]

        # Random contiguous character window so we sample uniformly across the doc
        if len(text) > CHAR_WINDOW:
            char_start = random.randint(0, len(text) - CHAR_WINDOW)
            text = text[char_start : char_start + CHAR_WINDOW]

        ids = self.tokenizer(text, truncation=False, padding=False)["input_ids"]
        n = len(ids)

        span_len = random.randint(min(MIN_SENT_LEN, n), min(n, MAX_SENT_LEN))
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
        raise RuntimeError("COPY_TO_TMPDIR=1 but SLURM_TMPDIR is not set")

    src_cache_dir = os.environ.get("FINEWEB_CACHE_DIR")
    if src_cache_dir is None:
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        src_cache_dir = os.path.join(hf_home, "datasets")

    sentinel = os.path.join(slurm_tmpdir, ".fineweb_copy_done")

    # Atomically claim the copy role: whichever process creates the sentinel
    # file first (O_EXCL is atomic on POSIX) does the copy; all others wait.
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


def load_fineweb() -> object:
    """Load FineWeb sample-100BT from local HF cache."""
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


def main():
    if os.environ.get("COPY_TO_TMPDIR", "0") == "1":
        _copy_dataset_to_tmpdir()

    hf_dataset = load_fineweb()

    print("Loading decoder...", flush=True)
    decoder = SentenceDecoder(DECODER_MODEL, LATENT_SIZE, MAX_SENT_LEN, device=DEVICE, device_map="auto")
    print("Loading encoder...", flush=True)
    encoder = SentenceEncoder(ENCODER_MODEL, LATENT_SIZE, decoder.tokenizer, device=DEVICE)

    train_dataset = FineWebSpanDataset(hf_dataset, decoder.tokenizer, size=NUM_TRAIN)
    eval_dataset = FineWebSpanDataset(hf_dataset, decoder.tokenizer, size=NUM_EVAL)

    model = LangVAE(VAEConfig(latent_dim=LATENT_SIZE), encoder, decoder)

    training_config = CyclicalScheduleKLThresholdTrainerConfig(
        output_dir=EXP_LABEL,
        num_epochs=NUM_EPOCHS,
        learning_rate=1e-3,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        train_dataloader_num_workers=NUM_DATALOADER_WORKERS,
        eval_dataloader_num_workers=NUM_DATALOADER_WORKERS,
        steps_saving=5,
        optimizer_cls="AdamW",
        scheduler_cls="ReduceLROnPlateau",
        scheduler_params={"patience": 5, "factor": 0.5},
        max_beta=1.0,
        n_cycles=N_CYCLES,
        target_kl=TARGET_KL,
    )

    pipeline = LanguageTrainingPipeline(training_config=training_config, model=model)
    tb_callback = TensorBoardCallback(EXP_LABEL)
    pipeline(train_data=train_dataset, eval_data=eval_dataset, callbacks=[tb_callback])


if __name__ == "__main__":
    main()
