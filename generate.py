"""
Generation script for Drifting Models.

Generates images from a trained drifting model with support for:
- Single-step generation (1 NFE)
- Classifier-free guidance (CFG)
- Class-conditional and unconditional generation
- Latent and pixel space modes

Usage:
  # Generate with default CFG
  python generate.py --ckpt outputs/drifting-dit-l2-latent/checkpoints/final.pt \
                     --cfg-scale 1.0 --num-samples 50000

  # Generate specific classes
  python generate.py --ckpt checkpoints/final.pt --classes 88,270,388

  # Generate for FID evaluation (50K samples)
  python generate.py --ckpt checkpoints/final.pt --num-samples 50000 \
                     --output-dir fid_samples --cfg-scale 1.0
"""

import os
import sys
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.utils import save_image
from tqdm import tqdm

from models.dit import DiT_models


@torch.no_grad()
def generate(
    model: nn.Module,
    vae=None,
    device: torch.device = torch.device("cuda"),
    num_samples: int = 50000,
    batch_size: int = 64,
    num_classes: int = 1000,
    cfg_scale: float = 1.0,
    latent_space: bool = True,
    in_channels: int = 4,
    input_size: int = 32,
    output_dir: str = "generated_samples",
    classes: list = None,
    save_grid: bool = True,
    save_individual: bool = True,
    seed: int = 0,
):
    """
    Generate images from a trained drifting model.

    Single forward pass (1 NFE) — the defining feature of drifting models.

    Args:
        model: Trained generator
        vae: VAE decoder (for latent space mode)
        device: Device
        num_samples: Total number of samples to generate
        batch_size: Batch size for generation
        num_classes: Number of classes
        cfg_scale: CFG scale (1.0 = no guidance)
        latent_space: Whether model operates in latent space
        in_channels: Input noise channels
        input_size: Spatial size of noise
        output_dir: Where to save generated images
        classes: Specific classes to generate (None = random)
        save_grid: Whether to save image grids
        save_individual: Whether to save individual images (for FID)
        seed: Random seed
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    num_generated = 0
    all_images = []
    all_labels = []

    pbar = tqdm(total=num_samples, desc="Generating")

    while num_generated < num_samples:
        current_batch = min(batch_size, num_samples - num_generated)

        # Generate class labels
        if classes is not None:
            # Cycle through specified classes
            label_idx = torch.arange(current_batch) % len(classes)
            labels = torch.tensor([classes[i] for i in label_idx], device=device)
        else:
            labels = torch.randint(0, num_classes, (current_batch,), device=device)

        # Generate noise
        noise = torch.randn(
            current_batch, in_channels, input_size, input_size,
            device=device,
        )

        # Forward pass with optional CFG
        if cfg_scale != 1.0:
            # Double batch: [conditional; unconditional]
            uncond_labels = torch.full_like(labels, num_classes)
            all_labels_batch = torch.cat([labels, uncond_labels])
            all_noise = torch.cat([noise, noise])

            output = model(all_noise, all_labels_batch)
            cond_out, uncond_out = output.chunk(2, dim=0)
            output = uncond_out + cfg_scale * (cond_out - uncond_out)
        else:
            output = model(noise, labels)

        # Decode from latent if needed
        if latent_space and vae is not None:
            images = vae.decode(output)
        else:
            images = output

        # Normalize to [0, 1]
        images = (images + 1) / 2
        images = images.clamp(0, 1)

        # Save individual images (for FID computation)
        if save_individual:
            for i in range(current_batch):
                img_path = os.path.join(output_dir, f"{num_generated + i:06d}.png")
                save_image(images[i], img_path)

        all_images.append(images.cpu())
        all_labels.append(labels.cpu())

        num_generated += current_batch
        pbar.update(current_batch)

    pbar.close()

    # Save a grid of samples
    if save_grid and len(all_images) > 0:
        grid_images = torch.cat(all_images, dim=0)[:64]
        grid_path = os.path.join(output_dir, "sample_grid.png")
        save_image(grid_images, grid_path, nrow=8, padding=2)
        print(f"Saved sample grid to {grid_path}")

    # Save labels
    all_labels = torch.cat(all_labels, dim=0)
    torch.save(all_labels, os.path.join(output_dir, "labels.pt"))

    print(f"Generated {num_generated} images in {output_dir}")
    return num_generated


def load_model_from_checkpoint(ckpt_path, device, use_ema=True):
    """
    Load a trained model from checkpoint.

    Args:
        ckpt_path: Path to checkpoint file
        device: Device to load to
        use_ema: Whether to use EMA weights (recommended)

    Returns:
        (model, config, vae)
    """
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    # Build model
    model = DiT_models[config["model"]](
        input_size=config["input_size"],
        in_channels=config["in_channels"],
        num_classes=config["num_classes"],
        class_dropout_prob=0.0,  # No dropout at inference
    ).to(device)

    # Load weights
    if use_ema and "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
        print("Loaded EMA weights")
    else:
        model.load_state_dict(ckpt["model"])
        print("Loaded model weights")

    model.eval()

    # Load VAE if needed
    vae = None
    if config.get("latent_space", True):
        from models.vae import AutoencoderKLWrapper
        vae = AutoencoderKLWrapper(
            pretrained_path=config.get("vae_path", "stabilityai/sd-vae-ft-mse"),
        ).to(device)
        vae.eval()

    return model, config, vae


def main():
    parser = argparse.ArgumentParser(description="Generate images from Drifting Model")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--output-dir", type=str, default="generated_samples")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="CFG scale (1.0 = no guidance)")
    parser.add_argument("--classes", type=str, default=None, help="Comma-separated class IDs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-ema", action="store_true", help="Don't use EMA weights")
    parser.add_argument("--save-grid-only", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, config, vae = load_model_from_checkpoint(
        args.ckpt, device, use_ema=not args.no_ema
    )

    # Parse classes
    classes = None
    if args.classes:
        classes = [int(c) for c in args.classes.split(",")]

    # Generate
    generate(
        model=model,
        vae=vae,
        device=device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_classes=config["num_classes"],
        cfg_scale=args.cfg_scale,
        latent_space=config.get("latent_space", True),
        in_channels=config["in_channels"],
        input_size=config["input_size"],
        output_dir=args.output_dir,
        classes=classes,
        save_grid=True,
        save_individual=not args.save_grid_only,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
