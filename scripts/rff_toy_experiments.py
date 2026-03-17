"""
RFF-Drift Toy Experiments.

Validates the research hypothesis "Accelerating Generative Drifting via Random
Fourier Feature Drift Estimation" on 2D synthetic benchmarks.

Six experiments from Section 6 of the theory paper, adapted to the toy setting:

  Exp 1 — Kernel Approximation Quality    (Theorem 4.2: O(1/√D) convergence rate)
  Exp 2 — Error Budget Comparison         (Theorem 4.3: RFF bias ≤ minibatch variance)
  Exp 3 — Direction Preservation          (cosine similarity of drift vectors)
  Exp 4 — Wall-Clock Speedup              (timing: exact vs RFF, varying d and B)
  Exp 5 — End-to-End Generation Quality   (train+MMD on 8-mode Gaussian ring)
  Exp 6 — Vector Field Visualization      (visual field comparison + energy dissipation)

Usage:
  # Run all experiments (~5-15 min on CPU):
  python scripts/rff_toy_experiments.py --all

  # Run a single experiment:
  python scripts/rff_toy_experiments.py --exp 1

  # Quick mode (fewer seeds/steps, for testing):
  python scripts/rff_toy_experiments.py --all --quick

  # Choose save directory:
  python scripts/rff_toy_experiments.py --all --save-dir toy_results/rff_experiments
"""

import os
import sys
import math
import time
import argparse
import csv
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

# Add parent directory to path so we can import from drifting/
sys.path.insert(0, str(Path(__file__).parent.parent))
from drifting.drift_field import compute_drift as compute_drift_exact
from drifting.rff_drift import compute_drift_rff, compute_drift_orf, sample_rff_params


# ─────────────────────────────────────────────────────────────
# Shared Dataset: 8-mode Gaussian Ring
# ─────────────────────────────────────────────────────────────

def sample_ring_mixture(n: int, n_modes: int = 8, radius: float = 0.8,
                        sigma: float = 0.08, seed: int = None) -> torch.Tensor:
    """
    Sample from an 8-mode Gaussian mixture arranged on a ring.

    Modes are equally spaced on a circle of the given radius. This is the
    canonical 2D benchmark from the paper (Section 6.6).

    Args:
        n: Number of samples
        n_modes: Number of modes (8 by default)
        radius: Ring radius
        sigma: Per-mode standard deviation
        seed: Optional random seed

    Returns:
        pts: [n, 2] float tensor
    """
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    angles = torch.linspace(0, 2 * math.pi, n_modes + 1)[:-1]
    centers = torch.stack([radius * torch.cos(angles),
                            radius * torch.sin(angles)], dim=1)   # [n_modes, 2]
    idx = torch.randint(0, n_modes, (n,), generator=g)
    noise = torch.randn(n, 2, generator=g) * sigma
    return centers[idx] + noise


# ─────────────────────────────────────────────────────────────
# Tiny MLP Generator (same architecture as toy_demo.py)
# ─────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim: int = 32, hidden: int = 256, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ─────────────────────────────────────────────────────────────
# Shared Utilities
# ─────────────────────────────────────────────────────────────

def mmd_squared(x: torch.Tensor, y: torch.Tensor,
                bandwidth: float = 0.5) -> float:
    """
    Unbiased MMD² estimator with Gaussian kernel.

    Used to measure distance between generated and real distributions
    without requiring a classifier.
    """
    def k(a, b):
        dist2 = torch.cdist(a, b).pow(2)
        return (-dist2 / (2 * bandwidth ** 2)).exp()

    n, m = x.shape[0], y.shape[0]
    kxx = k(x, x)
    kxx.fill_diagonal_(0)
    kyy = k(y, y)
    kyy.fill_diagonal_(0)
    kxy = k(x, y)
    return (kxx.sum() / (n * (n - 1)) +
            kyy.sum() / (m * (m - 1)) -
            2 * kxy.mean()).item()


def kde_entropy_proxy(gen: torch.Tensor, pos: torch.Tensor,
                      bandwidth: float = 0.3) -> float:
    """
    Approximation of KL(q_kde || p_kde) as energy proxy.

    Uses the drifting field magnitude as a proxy: when the field is near zero,
    the distributions are close (energy dissipated).
    """
    with torch.no_grad():
        V = compute_drift_exact(gen, pos, temp=0.05, v_norm=False)
        return V.norm(dim=-1).pow(2).mean().item()


def drifting_loss_fn(gen, pos, drift_fn):
    """MSE(gen, stopgrad(gen + V)) drifting loss."""
    with torch.no_grad():
        V = drift_fn(gen, pos)
        target = (gen + V).detach()
    return F.mse_loss(gen, target)


def save_csv(path: str, headers: list, rows: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def fig_save(fig, path: str, dpi: int = 150):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────
# Experiment 1 — Kernel Approximation Quality
# ─────────────────────────────────────────────────────────────

def run_exp1(save_dir: str, quick: bool = False):
    """
    Verify Theorem 4.2: ||V - V_D|| / ||V|| ~ O(1/√D).

    Samples query points from the ring mixture, computes exact vs RFF drift
    fields, and plots relative error vs D on a log-log scale. The fitted
    slope should be ≈ -0.5.
    """
    print("\n=== Experiment 1: Kernel Approximation Quality ===")
    out_dir = os.path.join(save_dir, "exp1_convergence")
    os.makedirs(out_dir, exist_ok=True)

    n_query  = 50  if quick else 200
    n_pos    = 256 if quick else 1024
    n_neg    = 64  if quick else 256
    n_seeds  = 10  if quick else 50
    D_values = [8, 16, 32, 64, 128, 256, 512]

    device = torch.device("cpu")
    torch.manual_seed(0)

    query = sample_ring_mixture(n_query, seed=1).to(device)
    pos   = sample_ring_mixture(n_pos,   seed=2).to(device)
    neg   = sample_ring_mixture(n_neg,   seed=3).to(device)

    # Exact drift (Laplace kernel, no v_norm so magnitudes are comparable)
    with torch.no_grad():
        V_exact = compute_drift_exact(query, pos, temp=0.05,
                                      v_norm=False, neg=neg)   # [Q, 2]

    mean_errors, std_errors = [], []
    csv_rows = []

    for D in D_values:
        errors_seeds = []
        for seed in range(n_seeds):
            with torch.no_grad():
                V_rff = compute_drift_rff(query, pos, D=D, bandwidth=0.05,
                                          v_norm=False, neg=neg, seed=seed)
            rel_err = (V_rff - V_exact).norm(dim=-1) / V_exact.norm(dim=-1).clamp_min(1e-8)
            errors_seeds.append(rel_err.mean().item())

        mu, sigma = float(np.mean(errors_seeds)), float(np.std(errors_seeds))
        mean_errors.append(mu)
        std_errors.append(sigma)
        csv_rows.append([D, mu, sigma])
        print(f"  D={D:4d}: rel_error = {mu:.4f} ± {sigma:.4f}")

    # Fit slope on log-log
    log_D = np.log(D_values)
    log_e = np.log(mean_errors)
    slope, intercept = np.polyfit(log_D, log_e, 1)
    print(f"\n  Fitted log-log slope: {slope:.3f}  (theoretical: -0.5)")
    print(f"  PASS: {-0.7 <= slope <= -0.3}")

    # Save CSV
    save_csv(os.path.join(out_dir, "relative_error.csv"),
             ["D", "mean_rel_error", "std_rel_error"], csv_rows)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(D_values, mean_errors, yerr=std_errors,
                fmt="o-", color="steelblue", capsize=4, label="RFF (Gaussian)")
    D_fit = np.array([D_values[0], D_values[-1]], dtype=float)
    ax.plot(D_fit, np.exp(intercept) * D_fit ** slope, "r--",
            label=f"Fitted slope {slope:.2f}")
    ax.plot(D_fit, np.exp(intercept) * D_fit ** (-0.5), "k:",
            label="Theoretical O(1/√D)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("RFF Dimension D")
    ax.set_ylabel("Relative Error ||V − V_D|| / ||V||")
    ax.set_title("Exp 1: RFF Convergence Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig_save(fig, os.path.join(out_dir, "convergence_rate.png"))

    return slope


# ─────────────────────────────────────────────────────────────
# Experiment 2 — Error Budget Comparison
# ─────────────────────────────────────────────────────────────

def run_exp2(save_dir: str, quick: bool = False):
    """
    Verify Theorem 4.3: RFF bias ≤ minibatch variance for D ≥ N.

    Computes a near-population drift (N=8192) as reference, then measures
    how much each error source contributes.
    """
    print("\n=== Experiment 2: Error Budget Comparison ===")
    out_dir = os.path.join(save_dir, "exp2_error_budget")
    os.makedirs(out_dir, exist_ok=True)

    n_query     = 20 if quick else 100
    n_pop       = 2048 if quick else 8192
    n_seeds     = 10 if quick else 50
    N_values    = [64, 128, 256, 512] if quick else [64, 128, 256, 512, 1024]
    D_ratios    = [0.25, 0.5, 1.0, 2.0]   # D = ratio × N

    device = torch.device("cpu")
    torch.manual_seed(0)

    query  = sample_ring_mixture(n_query, seed=10).to(device)
    neg_q  = sample_ring_mixture(256, seed=11).to(device)

    # Population drift (large N ≈ true expectation)
    pos_pop = sample_ring_mixture(n_pop, seed=12).to(device)
    with torch.no_grad():
        V_pop = compute_drift_exact(query, pos_pop, temp=0.05,
                                    v_norm=False, neg=neg_q)   # [Q, 2]

    # Results: dict[N][ratio] = MSE_total / Var_mb
    results = {N: {} for N in N_values}
    csv_rows = []

    for N in N_values:
        # Measure minibatch variance (exact kernel, vary N)
        var_mb_vals = []
        for seed in range(n_seeds):
            torch.manual_seed(seed * 7 + N)
            pos_mb = sample_ring_mixture(N, seed=seed * 7 + N).to(device)
            with torch.no_grad():
                V_mb = compute_drift_exact(query, pos_mb, temp=0.05,
                                           v_norm=False, neg=neg_q)
            mse = (V_mb - V_pop).norm(dim=-1).pow(2).mean().item()
            var_mb_vals.append(mse)
        var_mb = float(np.mean(var_mb_vals))

        for ratio in D_ratios:
            D = max(8, int(N * ratio))
            mse_total_vals = []
            for seed in range(n_seeds):
                torch.manual_seed(seed * 7 + N)
                pos_mb = sample_ring_mixture(N, seed=seed * 7 + N).to(device)
                with torch.no_grad():
                    V_rff = compute_drift_rff(query, pos_mb, D=D,
                                              bandwidth=0.05, v_norm=False,
                                              neg=neg_q, seed=seed * 13)
                mse = (V_rff - V_pop).norm(dim=-1).pow(2).mean().item()
                mse_total_vals.append(mse)
            mse_total = float(np.mean(mse_total_vals))
            ratio_val = mse_total / (var_mb + 1e-12)
            results[N][ratio] = ratio_val
            csv_rows.append([N, D, ratio, ratio_val, var_mb, mse_total])
            print(f"  N={N:4d}, D/N={ratio:.2f} (D={D:4d}): "
                  f"MSE/Var_mb = {ratio_val:.3f}")

    save_csv(os.path.join(out_dir, "error_budget.csv"),
             ["N", "D", "D_ratio", "mse_ratio", "var_mb", "mse_total"],
             csv_rows)

    # Plot 1: Line plot — ratio vs D/N for each N
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = cm.viridis(np.linspace(0, 0.8, len(N_values)))
    for color, N in zip(colors, N_values):
        ratios = [results[N][r] for r in D_ratios]
        ax.plot(D_ratios, ratios, "o-", color=color, label=f"N={N}")
    ax.axhline(y=1.0, color="gray", linestyle="--", label="MSE = Var_mb")
    ax.axhline(y=2.0, color="red",  linestyle=":",  label="2× Var_mb (pass threshold)")
    ax.set_xlabel("D/N ratio")
    ax.set_ylabel("MSE_total / Var_mb")
    ax.set_title("Exp 2: RFF Error vs Mini-Batch Variance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig_save(fig, os.path.join(out_dir, "error_budget_lines.png"))

    # Plot 2: Heatmap
    ratio_matrix = np.array([[results[N][r] for r in D_ratios]
                               for N in N_values])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(ratio_matrix, aspect="auto", vmin=0, vmax=4, cmap="RdYlGn_r")
    ax.set_xticks(range(len(D_ratios)))
    ax.set_xticklabels([str(r) for r in D_ratios])
    ax.set_yticks(range(len(N_values)))
    ax.set_yticklabels([str(N) for N in N_values])
    ax.set_xlabel("D/N ratio")
    ax.set_ylabel("Mini-batch size N")
    ax.set_title("Exp 2: MSE_total / Var_mb")
    plt.colorbar(im, ax=ax, label="Ratio")
    for i, N in enumerate(N_values):
        for j, r in enumerate(D_ratios):
            ax.text(j, i, f"{ratio_matrix[i,j]:.2f}", ha="center",
                    va="center", fontsize=8, color="black")
    fig_save(fig, os.path.join(out_dir, "error_budget_heatmap.png"))

    return results


# ─────────────────────────────────────────────────────────────
# Experiment 3 — Direction Preservation
# ─────────────────────────────────────────────────────────────

def run_exp3(save_dir: str, quick: bool = False):
    """
    Verify that RFF preserves the direction of the drift field.

    Measures cosine similarity cos(V, V_D) across D values.
    The directional accuracy is more important than magnitude
    because v_norm normalizes the drift step size.
    """
    print("\n=== Experiment 3: Direction Preservation ===")
    out_dir = os.path.join(save_dir, "exp3_direction")
    os.makedirs(out_dir, exist_ok=True)

    n_query  = 50  if quick else 200
    n_pos    = 256 if quick else 1024
    n_neg    = 64  if quick else 256
    n_seeds  = 10  if quick else 50
    D_values = [8, 16, 32, 64, 128, 256, 512]

    device = torch.device("cpu")
    torch.manual_seed(0)

    query = sample_ring_mixture(n_query, seed=20).to(device)
    pos   = sample_ring_mixture(n_pos,   seed=21).to(device)
    neg   = sample_ring_mixture(n_neg,   seed=22).to(device)

    with torch.no_grad():
        V_exact = compute_drift_exact(query, pos, temp=0.05,
                                      v_norm=False, neg=neg)
        V_norm  = V_exact / V_exact.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    mean_cos, std_cos = [], []
    csv_rows = []

    for D in D_values:
        cos_seeds = []
        for seed in range(n_seeds):
            with torch.no_grad():
                V_rff = compute_drift_rff(query, pos, D=D, bandwidth=0.05,
                                          v_norm=False, neg=neg, seed=seed)
                V_rff_n = V_rff / V_rff.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                cos_sim = (V_norm * V_rff_n).sum(dim=-1).mean().item()
            cos_seeds.append(cos_sim)

        mu, sigma = float(np.mean(cos_seeds)), float(np.std(cos_seeds))
        mean_cos.append(mu)
        std_cos.append(sigma)
        csv_rows.append([D, mu, sigma])
        print(f"  D={D:4d}: cosine similarity = {mu:.4f} ± {sigma:.4f}")

    save_csv(os.path.join(out_dir, "cosine_similarity.csv"),
             ["D", "mean_cosine", "std_cosine"], csv_rows)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(D_values, mean_cos, yerr=std_cos,
                fmt="o-", color="steelblue", capsize=4, label="RFF (Gaussian)")
    for y, label, style in [(0.90, "0.90 (pass)", "--"),
                             (0.95, "0.95",        ":"),
                             (0.99, "0.99",        "-.")]:
        ax.axhline(y=y, color="gray", linestyle=style, alpha=0.7, label=label)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("RFF Dimension D")
    ax.set_ylabel("Mean Cosine Similarity cos(V, V_D)")
    ax.set_title("Exp 3: Drift Direction Preservation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig_save(fig, os.path.join(out_dir, "cosine_similarity.png"))

    d64_idx = D_values.index(64)
    print(f"\n  PASS (cosine > 0.9 at D=64): {mean_cos[d64_idx] > 0.9}")
    return mean_cos


# ─────────────────────────────────────────────────────────────
# Experiment 4 — Wall-Clock Speedup
# ─────────────────────────────────────────────────────────────

def run_exp4(save_dir: str, quick: bool = False):
    """
    Measure actual wall-clock speedup of RFF vs exact drift computation.

    Uses synthetic feature vectors of varying dimension d to simulate
    the real use case (the 2D toy distribution won't show speedup since
    d=2; high-d vectors confirm the theoretical prediction).
    """
    print("\n=== Experiment 4: Wall-Clock Speedup ===")
    out_dir = os.path.join(save_dir, "exp4_speedup")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cpu")

    d_values = [2, 64, 256] if quick else [2, 64, 256, 512]
    B_values = [128, 256, 512] if quick else [128, 256, 512, 1024]
    D_values = [64, 128] if quick else [64, 128, 256]
    n_warmup = 5  if quick else 20
    n_timed  = 10 if quick else 50

    csv_rows = []

    for d in d_values:
        for B in B_values:
            N = B   # symmetric setup
            P = B

            gen_data = torch.randn(B, d)
            pos_data = torch.randn(P, d)
            neg_data = torch.randn(N, d)

            # Time exact drift
            for _ in range(n_warmup):
                compute_drift_exact(gen_data, pos_data, temp=0.05,
                                    v_norm=False, neg=neg_data)
            times_exact = []
            for _ in range(n_timed):
                t0 = time.perf_counter()
                compute_drift_exact(gen_data, pos_data, temp=0.05,
                                    v_norm=False, neg=neg_data)
                times_exact.append(time.perf_counter() - t0)
            t_exact = float(np.median(times_exact)) * 1000  # ms

            for D in D_values:
                for _ in range(n_warmup):
                    compute_drift_rff(gen_data, pos_data, D=D, bandwidth=0.05,
                                      v_norm=False, neg=neg_data, seed=0)
                times_rff = []
                for _ in range(n_timed):
                    t0 = time.perf_counter()
                    compute_drift_rff(gen_data, pos_data, D=D, bandwidth=0.05,
                                      v_norm=False, neg=neg_data, seed=0)
                    times_rff.append(time.perf_counter() - t0)
                t_rff = float(np.median(times_rff)) * 1000

                speedup = t_exact / (t_rff + 1e-9)
                csv_rows.append([d, B, D, t_exact, t_rff, speedup])
                print(f"  d={d:3d}, B={B:4d}, D={D:3d}: "
                      f"exact={t_exact:.2f}ms, RFF={t_rff:.2f}ms, "
                      f"speedup={speedup:.2f}x")

    save_csv(os.path.join(out_dir, "speedup_results.csv"),
             ["d", "B", "D", "t_exact_ms", "t_rff_ms", "speedup"],
             csv_rows)

    # Plot: speedup vs B for each d (at D = median D_values entry)
    D_plot = D_values[len(D_values) // 2]
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = cm.plasma(np.linspace(0.1, 0.9, len(d_values)))
    for color, d in zip(colors, d_values):
        speedups = [r[5] for r in csv_rows
                    if r[0] == d and r[2] == D_plot]
        Bs = [r[1] for r in csv_rows
              if r[0] == d and r[2] == D_plot]
        ax.plot(Bs, speedups, "o-", color=color, label=f"d={d}")
    ax.axhline(y=1.0, color="gray", linestyle="--", label="No speedup")
    ax.set_xlabel("Batch Size B (= N)")
    ax.set_ylabel(f"Speedup (exact / RFF) at D={D_plot}")
    ax.set_title("Exp 4: Wall-Clock Speedup")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig_save(fig, os.path.join(out_dir, "speedup_vs_batch.png"))

    # Heatmap: d × B for D_plot
    speedup_matrix = np.array(
        [[next((r[5] for r in csv_rows if r[0] == d and r[1] == B and r[2] == D_plot), 0)
          for B in B_values]
         for d in d_values]
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(speedup_matrix, aspect="auto", vmin=0, cmap="RdYlGn")
    ax.set_xticks(range(len(B_values)))
    ax.set_xticklabels([str(B) for B in B_values])
    ax.set_yticks(range(len(d_values)))
    ax.set_yticklabels([str(d) for d in d_values])
    ax.set_xlabel("Batch Size B")
    ax.set_ylabel("Feature Dimension d")
    ax.set_title(f"Speedup Heatmap (D={D_plot})")
    plt.colorbar(im, ax=ax, label="Speedup ×")
    for i in range(len(d_values)):
        for j in range(len(B_values)):
            ax.text(j, i, f"{speedup_matrix[i,j]:.1f}×",
                    ha="center", va="center", fontsize=8)
    fig_save(fig, os.path.join(out_dir, "speedup_heatmap.png"))

    return csv_rows


# ─────────────────────────────────────────────────────────────
# Experiment 5 — End-to-End Generation Quality
# ─────────────────────────────────────────────────────────────

def _train_one(method: str, D: int, n_steps: int, eval_every: int,
               seed: int, batch_size: int) -> dict:
    """Train one MLP generator and return loss + MMD curves."""
    device = torch.device("cpu")
    torch.manual_seed(seed)

    model = MLP(in_dim=32, hidden=256, out_dim=2).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Reference distribution (fixed, large)
    ref_gt = sample_ring_mixture(10000, seed=99).to(device)

    if method == "exact":
        drift_fn = partial(compute_drift_exact, temp=0.05, v_norm=True)
    elif method == "rff":
        drift_fn = partial(compute_drift_rff, D=D, bandwidth=0.05, v_norm=True)
    elif method == "orf":
        drift_fn = partial(compute_drift_orf, D=D, bandwidth=0.05, v_norm=True)
    else:
        raise ValueError(f"Unknown method: {method}")

    loss_history  = []
    mmd_steps     = []
    mmd_history   = []

    for step in range(1, n_steps + 1):
        pos = sample_ring_mixture(batch_size, seed=step).to(device)
        z   = torch.randn(batch_size, 32, device=device)
        gen = model(z)
        loss = drifting_loss_fn(gen, pos, drift_fn)

        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_history.append(loss.item())

        if step % eval_every == 0 or step == n_steps:
            with torch.no_grad():
                gen_eval = model(torch.randn(2000, 32, device=device))
            mmd_val = mmd_squared(gen_eval, ref_gt[:2000])
            mmd_steps.append(step)
            mmd_history.append(mmd_val)

    with torch.no_grad():
        final_gen = model(torch.randn(5000, 32, device=device)).cpu()

    return {
        "loss_history": loss_history,
        "mmd_steps":    mmd_steps,
        "mmd_history":  mmd_history,
        "final_gen":    final_gen,
        "final_loss":   float(np.mean(loss_history[-50:])),
        "final_mmd":    mmd_history[-1],
    }


def run_exp5(save_dir: str, quick: bool = False):
    """
    End-to-end generation quality comparison.

    Trains MLP generators on 8-mode Gaussian ring with exact Laplace kernel
    vs RFF-Gaussian kernels at various D. Evaluates MMD to held-out ground truth.
    """
    print("\n=== Experiment 5: End-to-End Generation Quality ===")
    out_dir = os.path.join(save_dir, "exp5_quality")
    os.makedirs(out_dir, exist_ok=True)

    n_seeds   = 2    if quick else 5
    n_steps   = 500  if quick else 3000
    eval_every = 50  if quick else 300
    batch_size = 256 if quick else 1024
    D_values  = [32, 64] if quick else [16, 32, 64, 128]

    methods_config = [("exact", None)] + [("rff", D) for D in D_values]

    all_results = {}  # method_key -> list of per-seed dicts

    gt_samples = sample_ring_mixture(5000, seed=999)

    for method, D in methods_config:
        key = "exact" if method == "exact" else f"rff_D{D}"
        label = "Exact (Laplace)" if method == "exact" else f"RFF D={D}"
        print(f"\n  Training: {label}  ({n_seeds} seeds)")
        seed_results = []

        for seed in range(n_seeds):
            print(f"    Seed {seed}...", end=" ", flush=True)
            t0 = time.time()
            res = _train_one(method, D, n_steps, eval_every,
                             seed=42 + seed, batch_size=batch_size)
            elapsed = time.time() - t0
            res["wall_time"] = elapsed
            seed_results.append(res)
            print(f"final MMD={res['final_mmd']:.5f}, "
                  f"loss={res['final_loss']:.4e}, "
                  f"time={elapsed:.1f}s")

        all_results[key] = seed_results

    # Save MMD CSV
    csv_rows = []
    for key, results in all_results.items():
        for seed_i, res in enumerate(results):
            for step, mmd in zip(res["mmd_steps"], res["mmd_history"]):
                csv_rows.append([key, seed_i, step, mmd])
    save_csv(os.path.join(out_dir, "mmd_curves.csv"),
             ["method", "seed", "step", "mmd"], csv_rows)

    # Summary table
    summary_rows = []
    for key, results in all_results.items():
        mmds  = [r["final_mmd"]  for r in results]
        times = [r["wall_time"]  for r in results]
        summary_rows.append([key,
                              f"{np.mean(mmds):.5f} ± {np.std(mmds):.5f}",
                              f"{np.mean(times):.1f}s"])
        print(f"  {key:<14}: final MMD = {np.mean(mmds):.5f} ± {np.std(mmds):.5f}")
    save_csv(os.path.join(out_dir, "summary.csv"),
             ["method", "final_mmd_mean±std", "wall_time_s"], summary_rows)

    # Pass check
    exact_mmd = np.mean([r["final_mmd"] for r in all_results["exact"]])
    for D in D_values:
        key = f"rff_D{D}"
        rff_mmd = np.mean([r["final_mmd"] for r in all_results[key]])
        passed = rff_mmd <= exact_mmd * 1.2
        print(f"  PASS D={D} (within 20% of exact): {passed}  "
              f"({rff_mmd:.5f} vs {exact_mmd:.5f})")

    # Plot 1: MMD curves
    colors = {"exact": "black"}
    palette = cm.viridis(np.linspace(0, 0.85, len(D_values)))
    for D, c in zip(D_values, palette):
        colors[f"rff_D{D}"] = c

    fig, ax = plt.subplots(figsize=(7, 4))
    for key, results in all_results.items():
        steps = results[0]["mmd_steps"]
        mmd_arr = np.array([r["mmd_history"] for r in results])
        mu = mmd_arr.mean(axis=0)
        se = mmd_arr.std(axis=0)
        label = "Exact (Laplace)" if key == "exact" else key.replace("rff_D", "RFF D=")
        lw = 2.5 if key == "exact" else 1.5
        ls = "-" if key == "exact" else "--"
        ax.plot(steps, mu, color=colors[key], lw=lw, ls=ls, label=label)
        ax.fill_between(steps, mu - se, mu + se,
                        alpha=0.15, color=colors[key])
    ax.set_xlabel("Training Step")
    ax.set_ylabel("MMD² to Ground Truth")
    ax.set_title("Exp 5: Generation Quality (MMD²)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig_save(fig, os.path.join(out_dir, "mmd_curves.png"))

    # Plot 2: Final scatter grid
    n_methods = len(all_results) + 1   # +1 for ground truth
    ncols = 3
    nrows = math.ceil(n_methods / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.2, nrows * 3.2))
    axes = axes.flatten()

    # Ground truth
    ax = axes[0]
    gt_np = gt_samples.numpy()
    ax.scatter(gt_np[:, 0], gt_np[:, 1], s=2, alpha=0.3, c="black")
    ax.set_title("Ground Truth", fontsize=9)
    ax.set_aspect("equal"); ax.axis("off")

    for i, (key, results) in enumerate(all_results.items()):
        ax = axes[i + 1]
        gen_np = results[0]["final_gen"].numpy()
        label = "Exact (Laplace)" if key == "exact" else key.replace("rff_D", "RFF D=")
        ax.scatter(gen_np[:, 0], gen_np[:, 1], s=2, alpha=0.3,
                   c=[colors[key]])
        mmd_val = np.mean([r["final_mmd"] for r in results])
        ax.set_title(f"{label}\nMMD²={mmd_val:.4f}", fontsize=9)
        ax.set_aspect("equal"); ax.axis("off")

    for j in range(i + 2, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Exp 5: Final Generated Distributions", fontsize=11, y=1.01)
    plt.tight_layout()
    fig_save(fig, os.path.join(out_dir, "final_scatter.png"))

    return all_results


# ─────────────────────────────────────────────────────────────
# Experiment 6 — Vector Field Visualization + Energy Dissipation
# ─────────────────────────────────────────────────────────────

def run_exp6(save_dir: str, quick: bool = False):
    """
    Visualize the RFF drift field vs exact field at training snapshots.
    Also plot the energy dissipation proxy over training.

    This is the most intuitive experiment: you can directly see whether the
    RFF field points in the same direction as the exact field.
    """
    print("\n=== Experiment 6: Vector Field Visualization + Energy Dissipation ===")
    out_dir = os.path.join(save_dir, "exp6_visualization")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cpu")
    torch.manual_seed(0)

    n_steps   = 500  if quick else 3000
    snap_steps = [0, 100, 300] if quick else [0, 500, 1000, 2000, 3000]
    D_values  = [8, 32] if quick else [8, 32, 128]
    batch_size = 256 if quick else 1024
    grid_res   = 12  if quick else 20

    # Grid of query points for field visualization
    lin = torch.linspace(-1.2, 1.2, grid_res)
    gx, gy = torch.meshgrid(lin, lin, indexing="ij")
    grid_pts = torch.stack([gx.flatten(), gy.flatten()], dim=1)   # [G², 2]

    # Build reference dataset
    pos_ref = sample_ring_mixture(1024, seed=50).to(device)

    # Train a model and collect snapshots
    model = MLP(in_dim=32, hidden=256, out_dim=2).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    drift_fn = partial(compute_drift_exact, temp=0.05, v_norm=True)

    # Store: step → dict of {method: V_field}
    snapshots = {}
    energy_log = {"exact": [], "steps": []}
    for D in D_values:
        energy_log[f"rff_D{D}"] = []

    for step in range(n_steps + 1):
        # Collect snapshot
        if step in snap_steps:
            with torch.no_grad():
                gen_snap = model(torch.randn(1024, 32, device=device))
                pos_snap = sample_ring_mixture(1024, seed=step * 3 + 1).to(device)

                snap = {}
                # Exact field on grid
                V_exact_grid = compute_drift_exact(
                    grid_pts, pos_snap, temp=0.05, v_norm=False, neg=gen_snap)
                snap["exact"] = V_exact_grid.numpy()

                # RFF fields on grid
                for D in D_values:
                    V_rff_grid = compute_drift_rff(
                        grid_pts, pos_snap, D=D, bandwidth=0.05,
                        v_norm=False, neg=gen_snap, seed=42)
                    snap[f"rff_D{D}"] = V_rff_grid.numpy()

                snapshots[step] = {
                    "snap": snap,
                    "gen": gen_snap.numpy(),
                    "pos": pos_snap.numpy(),
                }

        # Energy proxy measurement
        if step % (10 if quick else 50) == 0:
            with torch.no_grad():
                gen_e = model(torch.randn(512, 32, device=device))
                pos_e = sample_ring_mixture(512, seed=step + 100).to(device)
                energy_log["steps"].append(step)

                V_e = compute_drift_exact(gen_e, pos_e, temp=0.05, v_norm=False)
                energy_log["exact"].append(V_e.norm(dim=-1).pow(2).mean().item())

                for D in D_values:
                    V_rff_e = compute_drift_rff(gen_e, pos_e, D=D,
                                                bandwidth=0.05, v_norm=False,
                                                seed=D)
                    energy_log[f"rff_D{D}"].append(
                        V_rff_e.norm(dim=-1).pow(2).mean().item())

        if step == 0:
            continue  # don't train on step 0

        pos = sample_ring_mixture(batch_size, seed=step).to(device)
        z   = torch.randn(batch_size, 32, device=device)
        gen = model(z)
        loss = drifting_loss_fn(gen, pos, drift_fn)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # ── Plot 1: Vector field visualization ──
    row_labels = ["Exact"] + [f"RFF D={D}" for D in D_values]
    n_rows = len(row_labels)
    n_cols = len(snap_steps)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.8, n_rows * 2.8))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    vmax = None  # will be set from exact field

    for col_i, step in enumerate(snap_steps):
        snap_data = snapshots[step]["snap"]
        gen_pts   = snapshots[step]["gen"]
        pos_pts   = snapshots[step]["pos"]

        for row_i, (label, key) in enumerate(
                zip(row_labels, ["exact"] + [f"rff_D{D}" for D in D_values])):
            ax = axes[row_i, col_i]
            V  = snap_data[key]   # [G², 2]
            mag = np.linalg.norm(V, axis=-1)

            if vmax is None:
                vmax = np.percentile(mag, 95) + 1e-8

            ax.scatter(pos_pts[:200, 0], pos_pts[:200, 1],
                       s=3, alpha=0.25, c="black", zorder=2)
            ax.scatter(gen_pts[:200, 0], gen_pts[:200, 1],
                       s=3, alpha=0.25, c="tab:orange", zorder=2)
            ax.quiver(grid_pts[:, 0].numpy(),
                      grid_pts[:, 1].numpy(),
                      V[:, 0], V[:, 1],
                      mag, cmap="coolwarm", alpha=0.8,
                      scale=None, width=0.004,
                      clim=(0, vmax), zorder=3)

            ax.set_xlim(-1.4, 1.4)
            ax.set_ylim(-1.4, 1.4)
            ax.set_aspect("equal")
            ax.axis("off")

            if col_i == 0:
                ax.set_ylabel(label, fontsize=9)
            if row_i == 0:
                ax.set_title(f"Step {step}", fontsize=9)

    plt.suptitle("Exp 6: Drift Vector Fields\n"
                 "(black=real, orange=generated)", fontsize=10, y=1.01)
    plt.tight_layout()
    fig_save(fig, os.path.join(out_dir, "vector_fields.png"), dpi=120)

    # ── Plot 2: Energy dissipation ──
    fig, ax = plt.subplots(figsize=(7, 4))
    steps_arr = energy_log["steps"]
    ax.plot(steps_arr, energy_log["exact"],
            color="black", lw=2.5, label="Exact (Laplace)")
    colors_D = cm.viridis(np.linspace(0, 0.85, len(D_values)))
    for color, D in zip(colors_D, D_values):
        ax.plot(steps_arr, energy_log[f"rff_D{D}"],
                color=color, lw=1.5, ls="--", label=f"RFF D={D}")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Energy Proxy E[||V(gen)||²]")
    ax.set_title("Exp 6: Energy Dissipation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig_save(fig, os.path.join(out_dir, "energy_dissipation.png"))

    # Save energy CSV
    csv_rows = []
    for i, step in enumerate(steps_arr):
        row = [step, energy_log["exact"][i]]
        for D in D_values:
            row.append(energy_log[f"rff_D{D}"][i])
        csv_rows.append(row)
    save_csv(os.path.join(out_dir, "energy_dissipation.csv"),
             ["step", "exact"] + [f"rff_D{D}" for D in D_values],
             csv_rows)

    return snapshots, energy_log


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RFF-Drift Toy Experiments (Exps 1–6)")
    parser.add_argument("--exp", type=int, choices=[1, 2, 3, 4, 5, 6],
                        help="Run a single experiment (1–6)")
    parser.add_argument("--all", action="store_true",
                        help="Run all 6 experiments sequentially")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer seeds/steps for testing")
    parser.add_argument("--save-dir", type=str,
                        default="toy_results/rff_experiments",
                        help="Root directory for saving outputs")
    args = parser.parse_args()

    if not args.exp and not args.all:
        parser.print_help()
        return

    save_dir = args.save_dir
    q = args.quick

    if args.quick:
        print("[QUICK MODE] Using reduced configs for testing.\n")

    exp_fns = {
        1: lambda: run_exp1(save_dir, q),
        2: lambda: run_exp2(save_dir, q),
        3: lambda: run_exp3(save_dir, q),
        4: lambda: run_exp4(save_dir, q),
        5: lambda: run_exp5(save_dir, q),
        6: lambda: run_exp6(save_dir, q),
    }

    to_run = list(range(1, 7)) if args.all else [args.exp]

    t_start = time.time()
    for exp_num in to_run:
        exp_fns[exp_num]()

    total = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"All done. Total time: {total:.1f}s")
    print(f"Results saved to: {save_dir}/")


if __name__ == "__main__":
    main()
