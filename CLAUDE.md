# LangVAE — FineWeb Sentence VAE

Fork of [LangVAE](https://github.com/neuro-symbolic-ai/LangVAE) adapted to train a general-purpose sentence VAE on FineWeb, for use as a drop-in replacement for the frozen SentenceT5-XL encoder in [STAR-LDM](../ldm).

## What was changed

- **`langvae/encoders/automodel_presets.py`**: added `sentence-transformers/sentence-t5-xl` to `AUTOMODEL_MAP` (mean pooling + L2 normalization via `AutoModel`).
- **`examples/train_fineweb.py`**: training script using Flan-T5-base encoder + LLaMA-3.2-3B decoder, latent dim 768, trained on FineWeb `sample-100BT`.
- **`requirements.txt`**: added `datasets>=2.14` (FineWeb) and `sentencepiece>=0.2.0` (T5 tokenizer).

## Motivation

STAR-LDM uses frozen SentenceT5-XL to encode continuations into 768-dim embeddings, then runs diffusion in that space normalized by per-dimension mean/std statistics computed over FineWeb. Replacing the frozen encoder with a VAE trained on FineWeb gives:

- A proper Gaussian prior — no more empirical normalization hack
- A latent space regularized for diffusion to sample from directly

`LATENT_SIZE=768` is chosen to match SentenceT5-XL's output dimension and STAR-LDM's existing score net / soft prompt generator dimensions, so those components need no architectural changes.

## Commands

```bash
# Install
uv sync

# Train — single GPU
python train.py

# Train — multi-GPU (local torchrun)
torchrun --nproc_per_node=4 train.py

# Train — single SLURM job (launcher auto-detected from CC_CLUSTER env var)
python train.py

# Train — explicit launcher
python train.py hydra/launcher=fir

# Train — copy FineWeb to SLURM_TMPDIR first
python train.py copy_dataset_to_tmpdir=true hydra/launcher=fir

# Hyperparameter sweep
python train.py --multirun latent_size=128,256 n_cycles=10,20 hydra/launcher=fir

# Override any config value inline
python train.py latent_size=128 target_kl=2.0 batch_size=32

# Monitor training
tensorboard --logdir=runs
```

Config lives in `configs/train.yaml`; override any key on the CLI.
SLURM launcher configs are in `configs/hydra/launcher/` (fir / nibi / tamia).
`CC_CLUSTER` env var auto-selects the matching launcher when set.

Checkpoints are saved to `fineweb-langvae-<encoder>-<decoder>-l<latent_size>/`.

## Key hyperparameters

| Param | Value | Rationale |
|---|---|---|
| `LATENT_SIZE` | 768 | Matches SentenceT5-XL output and STAR-LDM dims |
| `MAX_SENT_LEN` | 128 | Matches STAR-LDM's `MAX_LENGTH` |
| `target_kl` | 12.0 | Default 2.0 scaled by latent_size ratio (768/128) |
| `n_cycles` | 40 | KL annealing cycles over full training run |
| `NUM_TRAIN` | 5_000_000 | Steps per epoch (docs sampled with replacement) |
| `NUM_EVAL` | 50_000 | Eval steps per epoch |
| `NUM_EPOCHS` | 5 | Training epochs |
| `BATCH_SIZE` | 64 | Per device; effective batch 256 on 4 GPUs |
| `CHAR_WINDOW` | 1536 | Character window drawn from doc before tokenizing |
| `MIN_SENT_LEN` | 16 | Minimum span length in tokens |

`target_kl` is a rough heuristic and the most important hyperparameter to tune if the latent space is underutilized (KL collapses to near zero) or over-regularized (reconstruction quality is poor).

## Integrating the trained VAE into STAR-LDM

The trained VAE encoder replaces the frozen `sentence_encoder` in `TransfusionGPT`. At inference use the posterior mean (not a sample) for determinism. The encoder internally handles re-tokenization from the LLaMA token space to the SentenceT5 token space.

```python
from langvae import LangVAE

vae = LangVAE.load_from_folder("fineweb-langvae-.../final_model")
vae.encoder.to(device)
vae.encoder.init_pretrained_model()

# tok_ids: LLaMA-tokenized continuation, shape (B, S)
z = vae.encoder.forward(tok_ids).embedding  # posterior mean, shape (B, 768)
```

Drop the per-dimension mean/std normalization in STAR-LDM (`data_stats/fineweb_100b/`) — the VAE prior regularizes `z` toward Gaussian(0,1) directly.
