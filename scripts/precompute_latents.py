"""
Precompute VAE latents for ImageNet.

This script encodes the entire ImageNet training set through the SD-VAE
and saves the latent representations. This speeds up latent-space training
by avoiding redundant VAE encoding at each training step.

Usage:
  python scripts/precompute_latents.py \
    --data-path /path/to/imagenet \
    --output-dir data/imagenet_latents \
    --batch-size 128
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.vae import AutoencoderKLWrapper
from data.imagenet import ImageNetDataset


def main():
    parser = argparse.ArgumentParser(description="Precompute VAE latents")
    parser.add_argument("--data-path", type=str, required=True, help="ImageNet root")
    parser.add_argument("--output-dir", type=str, default="data/imagenet_latents")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--vae-path", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKLWrapper(pretrained_path=args.vae_path).to(device)

    # Load dataset
    print("Loading dataset...")
    dataset = ImageNetDataset(
        root=args.data_path,
        image_size=args.image_size,
        split="train",
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Encode
    print(f"Encoding {len(dataset)} images...")
    all_latents = []
    all_labels = []

    for images, labels in tqdm(loader):
        images = images.to(device)
        with torch.no_grad():
            latents = vae.encode(images)
        all_latents.append(latents.cpu())
        all_labels.append(labels)

    all_latents = torch.cat(all_latents, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Save
    print(f"Saving latents: {all_latents.shape}")
    torch.save(all_latents, os.path.join(args.output_dir, "latents.pt"))
    torch.save(all_labels, os.path.join(args.output_dir, "labels.pt"))
    print(f"Saved to {args.output_dir}")
    print(f"  Latents: {all_latents.shape} ({all_latents.element_size() * all_latents.nelement() / 1e9:.1f} GB)")
    print(f"  Labels: {all_labels.shape}")


if __name__ == "__main__":
    main()
