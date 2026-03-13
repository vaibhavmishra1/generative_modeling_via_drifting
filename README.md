# Generative Modeling via Drifting

Implementation of **"Generative Modeling via Drifting"** (Deng, Li, Li, Du, He — MIT/Harvard, 2025).

> **Paper:** [arXiv:2602.04770](https://arxiv.org/pdf/2602.04770)
> **Project page:** [lambertae.github.io/projects/drifting](https://lambertae.github.io/projects/drifting/)

## Overview

Drifting Models are a new paradigm for generative modeling that evolves the pushforward distribution during **training time** rather than inference time (as in diffusion/flow models). This enables **single-step generation (1 NFE)** while achieving state-of-the-art results.

**Key results on ImageNet 256×256:**
| Setting | FID ↓ | IS ↑ | Params | NFE |
|---------|-------|------|--------|-----|
| Latent (SD-VAE) | **1.54** | 258.9 | 463M | 1 |
| Pixel space | **1.61** | 307.5 | 464M | 1 |

## Core Algorithm

The training objective is:
```
L = E_ε[‖f_θ(ε) - stopgrad(f_θ(ε) + V_{p,q}(f_θ(ε)))‖²]
```

Where the **drifting field** V_{p,q} uses a kernel-based mean-shift formulation:
```
V_{p,q}(x) = (1/Z_p Z_q) E[k(x,y⁺)·k(x,y⁻)·(y⁺ - y⁻)]
```
- `y⁺` ~ data distribution (attraction)
- `y⁻` ~ generated distribution (repulsion)
- V = 0 at equilibrium (when distributions match)

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run toy 2D demo (validates core algorithm)
```bash
python scripts/toy_demo.py --distribution swiss_roll --steps 2000
python scripts/toy_demo.py --distribution checkerboard --steps 2000
```

### 3. Train on ImageNet 256×256

**Latent space (recommended — FID 1.54):**
```bash
# Edit data_path in config first
vim configs/latent_dit_l2.yaml

# Single GPU
python train.py --config configs/latent_dit_l2.yaml --data-path /path/to/imagenet

# Multi-GPU (8 GPUs)
torchrun --nproc_per_node=8 train.py --config configs/latent_dit_l2.yaml --data-path /path/to/imagenet

# Or use the launch script
bash scripts/launch_train.sh configs/latent_dit_l2.yaml 8 /path/to/imagenet
```

**Pixel space (FID 1.61):**
```bash
torchrun --nproc_per_node=8 train.py --config configs/pixel_dit_l2.yaml --data-path /path/to/imagenet
```

### 4. Optional: Precompute VAE latents (faster training)
```bash
python scripts/precompute_latents.py --data-path /path/to/imagenet --output-dir data/imagenet_latents
```

### 5. Generate samples
```bash
# Generate 50K samples for FID evaluation
python generate.py --ckpt outputs/drifting-latent-dit-l2/checkpoints/final.pt \
                   --num-samples 50000 --cfg-scale 1.0

# Generate specific classes with CFG
python generate.py --ckpt checkpoints/final.pt --classes 88,270,388 \
                   --cfg-scale 1.5 --num-samples 64
```

### 6. Evaluate FID
```bash
python evaluate.py --ckpt checkpoints/final.pt --ref-path /path/to/imagenet/train \
                   --num-samples 50000 --cfg-scale 1.0 --compute-is
```

## Project Structure

```
├── configs/
│   ├── latent_dit_l2.yaml    # Latent-space config (main result)
│   └── pixel_dit_l2.yaml     # Pixel-space config
├── models/
│   ├── dit.py                # Modified DiT (no timestep, class-only)
│   ├── vae.py                # SD-VAE wrapper
│   └── feature_encoder.py    # MAE/DINOv2 feature encoder
├── drifting/
│   ├── drift_field.py        # Drifting field V computation
│   ├── loss.py               # Multi-scale drifting loss
│   └── queue.py              # Sample queue system
├── data/
│   └── imagenet.py           # ImageNet dataset & loader
├── scripts/
│   ├── toy_demo.py           # Toy 2D demo
│   ├── precompute_latents.py # Precompute VAE latents
│   └── launch_train.sh       # Multi-GPU launch script
├── train.py                  # Main training script
├── generate.py               # Sample generation
├── evaluate.py               # FID/IS evaluation
└── requirements.txt
```

## Key Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | DiT-L/2 | ~463M params, patch size 2 |
| Input size | 32×32 | Latent spatial (256/8) |
| Channels | 4 | SD-VAE latent channels |
| Feature encoder | MAE ViT-L/16 | Multi-scale features |
| Feature blocks | [7, 15, 23] | 1/3, 2/3, final layer |
| Kernel temp τ | 0.05 | Per-scale temperature |
| V-normalization | ✓ | Unit norm drift vectors |
| Queue (per-class) | 128 | Positive sample buffer |
| Queue (global) | 1000 | Unconditional buffer |
| Label dropout | 10% | For CFG training |
| Optimizer | AdamW | lr=1e-4, β=(0.9, 0.999) |
| EMA decay | 0.9999 | Exponential moving average |
| Total steps | 800K | Training iterations |
| Batch size | 256 | Global (32 per GPU × 8) |
| Mixed precision | bf16 | Automatic mixed precision |

## Hardware Requirements

- **Minimum:** 1× A100 80GB (reduced batch size)
- **Recommended:** 8× A100 80GB (matches paper setup)
- **Training time:** ~3-5 days on 8× A100

## Citation

```bibtex
@article{deng2025drifting,
  title={Generative Modeling via Drifting},
  author={Deng, Mingyang and Li, He and Li, Tianhong and Du, Yilun and He, Kaiming},
  journal={arXiv preprint arXiv:2602.04770},
  year={2025}
}
```

## Notes

- The official code has not been publicly released yet. This is a faithful reimplementation based on the paper's descriptions, algorithms, and stated configurations.
- Some hyperparameters (exact learning rate schedule, kernel bandwidth tuning) may require adjustment to exactly match the paper's reported numbers.
- Start with the toy demo to validate your setup before scaling to ImageNet.
