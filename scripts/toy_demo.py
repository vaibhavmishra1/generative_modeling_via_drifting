"""
Toy 2D demonstration of the Drifting Model algorithm.

Validates the core drifting algorithm on 2D distributions:
- Swiss Roll
- Checkerboard
- Gaussian Mixture

This is a minimal, self-contained version that can run on CPU
to verify the algorithm implementation before scaling to ImageNet.

Based on the official Colab demo from the paper's project page.

Usage:
  python scripts/toy_demo.py --distribution swiss_roll --steps 2000
  python scripts/toy_demo.py --distribution checkerboard --steps 2000
"""

import os
import sys
import math
import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────
# Toy Datasets
# ─────────────────────────────────────────────────────────────

def sample_checkerboard(n, noise=0.05, seed=None):
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    b = torch.randint(0, 2, (n,), generator=g)
    i = torch.randint(0, 2, (n,), generator=g) * 2 + b
    j = torch.randint(0, 2, (n,), generator=g) * 2 + b
    u = torch.rand(n, generator=g)
    v = torch.rand(n, generator=g)
    pts = torch.stack([i + u, j + v], dim=1) - 2.0
    pts = pts / 2.0
    if noise > 0:
        pts = pts + noise * torch.randn(pts.shape, generator=g)
    return pts


def sample_swiss_roll(n, noise=0.03, seed=None):
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    u = torch.rand(n, generator=g)
    t = 0.5 * math.pi + 4.0 * math.pi * u
    pts = torch.stack([t * torch.cos(t), t * torch.sin(t)], dim=1)
    pts = pts / (pts.abs().max() + 1e-8)
    if noise > 0:
        pts = pts + noise * torch.randn(pts.shape, generator=g)
    return pts


def sample_gaussian_mixture(n, seed=None):
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    centers = torch.tensor([
        [-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5],
        [0.0, 0.0],
    ])
    idx = torch.randint(0, len(centers), (n,), generator=g)
    pts = centers[idx] + torch.randn(n, 2, generator=g) * 0.08
    return pts


# ─────────────────────────────────────────────────────────────
# Core Algorithm (same as paper's demo)
# ─────────────────────────────────────────────────────────────

def compute_drift(gen, pos, temp=0.05):
    """Compute drift field V with batch-normalized kernel."""
    targets = torch.cat([gen, pos], dim=0)
    G = gen.shape[0]

    dist = torch.cdist(gen, targets)
    dist[:, :G].fill_diagonal_(1e6)
    kernel = (-dist / temp).exp()

    normalizer = kernel.sum(dim=-1, keepdim=True) * kernel.sum(dim=-2, keepdim=True)
    normalizer = normalizer.clamp_min(1e-12).sqrt()
    normalized_kernel = kernel / normalizer

    pos_coeff = normalized_kernel[:, G:] * normalized_kernel[:, :G].sum(dim=-1, keepdim=True)
    pos_V = pos_coeff @ targets[G:]
    neg_coeff = normalized_kernel[:, :G] * normalized_kernel[:, G:].sum(dim=-1, keepdim=True)
    neg_V = neg_coeff @ targets[:G]

    return pos_V - neg_V


def drifting_loss(gen, pos, compute_drift_fn):
    """Drifting loss: MSE(gen, stopgrad(gen + V))."""
    with torch.no_grad():
        V = compute_drift_fn(gen, pos)
        target = (gen + V).detach()
    return F.mse_loss(gen, target)


# ─────────────────────────────────────────────────────────────
# MLP Generator
# ─────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Simple MLP: noise → 2D output."""
    def __init__(self, in_dim=32, hidden=256, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z):
        return self.net(z)


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train(sampler, steps=2000, data_bs=2048, gen_bs=2048, lr=1e-3,
          temp=0.05, in_dim=32, hidden=256, plot_every=500, seed=42,
          save_dir="toy_results"):
    """Train drifting model on 2D distribution."""
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=in_dim, hidden=hidden, out_dim=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    ema = None
    drift_fn = partial(compute_drift, temp=temp)

    pbar = tqdm(range(1, steps + 1))
    for step in pbar:
        pos = sampler(data_bs).to(device)
        gen = model(torch.randn(gen_bs, in_dim, device=device))
        loss = drifting_loss(gen, pos, drift_fn)

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_history.append(loss.item())
        ema = loss.item() if ema is None else 0.96 * ema + 0.04 * loss.item()
        pbar.set_postfix(loss=f"{ema:.2e}")

        if step % plot_every == 0 or step == 1:
            with torch.no_grad():
                vis = model(torch.randn(5000, in_dim, device=device)).cpu().numpy()
                gt = sampler(5000).numpy()

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
            ax1.scatter(gt[:, 0], gt[:, 1], s=2, alpha=0.3, c="black")
            ax1.set_title("Target")
            ax1.set_aspect("equal")
            ax1.axis("off")

            ax2.scatter(vis[:, 0], vis[:, 1], s=2, alpha=0.3, c="tab:orange")
            ax2.set_title(f"Generated (step {step})")
            ax2.set_aspect("equal")
            ax2.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"step_{step:05d}.png"), dpi=150)
            plt.close()

    # Plot loss curve
    plt.figure(figsize=(6, 3))
    plt.plot(loss_history, alpha=0.7)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=150)
    plt.close()

    print(f"\nFinal loss: {loss_history[-1]:.4e}")
    print(f"Results saved to: {save_dir}")
    return model, loss_history


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Toy 2D Drifting Model Demo")
    parser.add_argument("--distribution", type=str, default="swiss_roll",
                        choices=["swiss_roll", "checkerboard", "gaussian_mixture"])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    samplers = {
        "swiss_roll": sample_swiss_roll,
        "checkerboard": sample_checkerboard,
        "gaussian_mixture": sample_gaussian_mixture,
    }

    sampler = samplers[args.distribution]
    save_dir = args.save_dir or f"toy_results/{args.distribution}"

    print(f"Training on: {args.distribution}")
    print(f"Steps: {args.steps}, Temp: {args.temp}, LR: {args.lr}")

    model, loss_history = train(
        sampler,
        steps=args.steps,
        temp=args.temp,
        lr=args.lr,
        seed=args.seed,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
