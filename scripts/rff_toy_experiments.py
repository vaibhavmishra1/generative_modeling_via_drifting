"""
RFF-Drift Toy Experiments (v2 — fixed kernel matching).

Validates the research hypothesis "Accelerating Generative Drifting via Random
Fourier Feature Drift Estimation" on 2D synthetic benchmarks.

**Critical design decision**: The exact baseline uses the *Laplace* kernel
  k(x,y) = exp(-||x-y||/τ).
The primary RFF comparison therefore uses Laplace RFF (Cauchy spectral
distribution) to approximate the *same* kernel.  Gaussian RFF is included as
a secondary comparison to demonstrate the kernel-mismatch failure mode.

Experiments:

  Exp 0 — Sanity Check            (verify Laplace RFF ≈ exact at high D)
  Exp 1 — Kernel Approx Quality   (Theorem 4.2: O(1/√D) convergence rate)
  Exp 2 — Error Budget Comparison  (Theorem 4.3: RFF bias ≤ minibatch variance)
  Exp 3 — Direction Preservation   (cosine similarity of drift vectors)
  Exp 4 — Wall-Clock Speedup       (timing: exact vs RFF, varying d and B)
  Exp 5 — End-to-End Quality       (train+MMD on 8-mode Gaussian ring)
  Exp 6 — Vector Field Vis.        (field comparison + energy dissipation)

Usage:
  python scripts/rff_toy_experiments.py --all
  python scripts/rff_toy_experiments.py --exp 0          # sanity check
  python scripts/rff_toy_experiments.py --all --quick    # smoke test (~2 min)
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

# Add parent directory to path so we can import from drifting/
sys.path.insert(0, str(Path(__file__).parent.parent))
from drifting.drift_field import compute_drift as compute_drift_exact
from drifting.rff_drift import compute_drift_rff, sample_rff_params, rff_map


# ─────────────────────────────────────────────────────────────
# Shared Dataset: 8-mode Gaussian Ring
# ─────────────────────────────────────────────────────────────

def sample_ring_mixture(n: int, n_modes: int = 8, radius: float = 0.8,
                        sigma: float = 0.08, seed: int = None) -> torch.Tensor:
    """
    Sample from an 8-mode Gaussian mixture arranged on a ring.

    Modes are equally spaced on a circle of the given radius. This is the
    canonical 2D benchmark from §6.6 of the theory paper.
    """
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    angles = torch.linspace(0, 2 * math.pi, n_modes + 1)[:-1]
    centers = torch.stack([radius * torch.cos(angles),
                            radius * torch.sin(angles)], dim=1)
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
                bandwidths: list = None) -> float:
    """
    Multi-bandwidth MMD² estimator (more robust than single bandwidth).

    Averages MMD² across several bandwidth values so the result is not
    sensitive to a single bandwidth choice.
    """
    if bandwidths is None:
        bandwidths = [0.1, 0.3, 0.5, 1.0]

    total = 0.0
    n, m = x.shape[0], y.shape[0]
    dist_xx = torch.cdist(x, x).pow(2)
    dist_yy = torch.cdist(y, y).pow(2)
    dist_xy = torch.cdist(x, y).pow(2)

    for bw in bandwidths:
        kxx = (-dist_xx / (2 * bw ** 2)).exp()
        kxx.fill_diagonal_(0)
        kyy = (-dist_yy / (2 * bw ** 2)).exp()
        kyy.fill_diagonal_(0)
        kxy = (-dist_xy / (2 * bw ** 2)).exp()
        total += (kxx.sum() / (n * (n - 1)) +
                  kyy.sum() / (m * (m - 1)) -
                  2 * kxy.mean()).item()
    return total / len(bandwidths)


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


# ── Kernel temperatures ──
# The codebase uses τ=0.05 for high-dimensional ViT features (d=1024).
# For 2D toy data, τ=0.05 is extremely sharp: the kernel is non-negligible
# only for ||x-y|| < 0.25.  The RFF noise floor (√(2/D) ≈ 0.09 at D=256)
# can swamp the batch normalization when most kernel values are < 0.01.
#
# We test TWO temperatures:
#   TEMP_WIDE  = 0.3  — moderate kernel, RFF noise floor << kernel values,
#                        clean O(1/√D) convergence expected.
#   TEMP_SHARP = 0.05 — original paper setting, tests RFF under stress.
TEMP_WIDE  = 0.3
TEMP_SHARP = 0.05


# ─────────────────────────────────────────────────────────────
# Experiment 0 — Sanity Check
# ─────────────────────────────────────────────────────────────

def run_exp0(save_dir: str, quick: bool = False):
    """
    Verify that Laplace RFF ≈ exact Laplace drift.

    Tests TWO temperatures (τ=0.3 wide, τ=0.05 sharp) to show the effect
    of the RFF noise floor on the batch-normalized drift.  Also shows
    Gaussian RFF as a kernel-mismatch control.
    """
    print("\n=== Experiment 0: Sanity Check (Laplace RFF ≈ Exact) ===")
    out_dir = os.path.join(save_dir, "exp0_sanity")
    os.makedirs(out_dir, exist_ok=True)

    n_query  = 50  if quick else 200
    n_pos    = 256 if quick else 1024
    n_neg    = 64  if quick else 256
    n_seeds  = 10  if quick else 30

    device = torch.device("cpu")
    query = sample_ring_mixture(n_query, seed=1).to(device)
    pos   = sample_ring_mixture(n_pos,   seed=2).to(device)
    neg   = sample_ring_mixture(n_neg,   seed=3).to(device)

    all_scatter_data = []

    for temp, temp_label in [(TEMP_WIDE, "τ=0.3 (wide)"),
                              (TEMP_SHARP, "τ=0.05 (sharp)")]:
        print(f"\n  --- {temp_label} ---")
        with torch.no_grad():
            V_exact = compute_drift_exact(query, pos, temp=temp,
                                          v_norm=False, neg=neg)
        print(f"  Exact drift mean magnitude: "
              f"{V_exact.norm(dim=-1).mean():.6f}")

        for kernel_name in ["laplace", "gaussian"]:
            D_high = 256 if quick else 1024
            errs, cos_sims = [], []
            for s in range(n_seeds):
                with torch.no_grad():
                    V_rff = compute_drift_rff(
                        query, pos, D=D_high, bandwidth=temp,
                        v_norm=False, neg=neg,
                        kernel=kernel_name, seed=s)
                rel = ((V_rff - V_exact).norm(dim=-1) /
                       V_exact.norm(dim=-1).clamp_min(1e-8))
                errs.append(rel.mean().item())
                cos = F.cosine_similarity(V_rff, V_exact, dim=-1).mean().item()
                cos_sims.append(cos)

            mu_err = np.mean(errs)
            mu_cos = np.mean(cos_sims)
            is_match = kernel_name == "laplace"
            passed = mu_cos > 0.7
            status = ("PASS" if (is_match and passed) else
                      "EXPECTED FAIL" if not is_match else "FAIL")
            print(f"    {kernel_name:>8} RFF D={D_high}: "
                  f"rel_err={mu_err:.3f}, cos={mu_cos:.3f}  [{status}]")

        # Collect data for scatter plot
        with torch.no_grad():
            V_lap = compute_drift_rff(
                query, pos, D=512, bandwidth=temp,
                v_norm=False, neg=neg, kernel="laplace", seed=0)
        all_scatter_data.append((temp_label, V_exact.clone(), V_lap.clone()))

    # Scatter plot: exact vs Laplace RFF at both temperatures
    fig, axes = plt.subplots(1, len(all_scatter_data),
                             figsize=(5.5 * len(all_scatter_data), 4.5))
    if len(all_scatter_data) == 1:
        axes = [axes]
    for ax, (label, V_ex, V_rf) in zip(axes, all_scatter_data):
        for dim, dl in [(0, "x"), (1, "y")]:
            ax.scatter(V_ex[:, dim].numpy(), V_rf[:, dim].numpy(),
                       s=10, alpha=0.5, label=f"dim {dl}")
        lim = max(V_ex.abs().max().item(), V_rf.abs().max().item()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, label="y=x")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("Exact drift")
        ax.set_ylabel("Laplace RFF D=512")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    plt.suptitle("Exp 0: Exact vs Laplace RFF Drift", y=1.02)
    plt.tight_layout()
    fig_save(fig, os.path.join(out_dir, "sanity_scatter.png"))


# ─────────────────────────────────────────────────────────────
# Experiment 1 — Kernel Approximation Quality
# ─────────────────────────────────────────────────────────────

def run_exp1(save_dir: str, quick: bool = False):
    """
    Verify Theorem 4.2: ||V - V_D|| / ||V|| ~ O(1/√D).

    Tests at TWO temperatures:
      τ=0.3  (wide)  — RFF noise floor ≪ kernel values → clean convergence
      τ=0.05 (sharp) — noise floor can dominate → slower / limited convergence
    """
    print("\n=== Experiment 1: Kernel Approximation Quality ===")
    out_dir = os.path.join(save_dir, "exp1_convergence")
    os.makedirs(out_dir, exist_ok=True)

    n_query  = 50  if quick else 200
    n_pos    = 256 if quick else 1024
    n_neg    = 64  if quick else 256
    n_seeds  = 10  if quick else 50
    D_values = [8, 16, 32, 64, 128, 256, 512]
    temps    = [TEMP_WIDE, TEMP_SHARP]

    device = torch.device("cpu")
    query = sample_ring_mixture(n_query, seed=1).to(device)
    pos   = sample_ring_mixture(n_pos,   seed=2).to(device)
    neg   = sample_ring_mixture(n_neg,   seed=3).to(device)

    all_slopes = {}
    all_csv_rows = []

    fig, axes = plt.subplots(1, len(temps), figsize=(7 * len(temps), 5))
    if len(temps) == 1:
        axes = [axes]

    for ax, temp in zip(axes, temps):
        tl = f"τ={temp}"
        print(f"\n  --- {tl} ---")
        with torch.no_grad():
            V_exact = compute_drift_exact(query, pos, temp=temp,
                                          v_norm=False, neg=neg)

        mean_errors, std_errors = [], []
        for D in D_values:
            errors_seeds = []
            for seed in range(n_seeds):
                with torch.no_grad():
                    V_rff = compute_drift_rff(
                        query, pos, D=D, bandwidth=temp, v_norm=False,
                        neg=neg, kernel="laplace", seed=seed)
                rel = ((V_rff - V_exact).norm(dim=-1) /
                       V_exact.norm(dim=-1).clamp_min(1e-8))
                errors_seeds.append(rel.mean().item())

            mu = float(np.mean(errors_seeds))
            sigma = float(np.std(errors_seeds))
            mean_errors.append(mu)
            std_errors.append(sigma)
            all_csv_rows.append([temp, D, mu, sigma])
            print(f"    D={D:4d}: rel_error = {mu:.4f} ± {sigma:.4f}")

        # Fit log-log slope
        log_D = np.log(D_values)
        log_e = np.log(np.clip(mean_errors, 1e-10, None))
        slope, intercept = np.polyfit(log_D, log_e, 1)
        all_slopes[temp] = slope
        print(f"    Fitted slope: {slope:.3f}  (theoretical: -0.5)")

        ax.errorbar(D_values, mean_errors, yerr=std_errors,
                    fmt="o-", color="steelblue", capsize=4,
                    label="Laplace RFF")
        D_fit = np.array([D_values[0], D_values[-1]], dtype=float)
        ax.plot(D_fit, np.exp(intercept) * D_fit ** slope, "r--",
                alpha=0.6, label=f"Fit: slope {slope:.2f}")
        ax.plot(D_fit, np.exp(intercept) * D_fit ** (-0.5), "k:",
                label="O(1/√D)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("RFF Dimension D")
        ax.set_ylabel("Relative Error")
        ax.set_title(f"{tl}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    save_csv(os.path.join(out_dir, "relative_error.csv"),
             ["temp", "D", "mean_rel_error", "std_rel_error"],
             all_csv_rows)
    plt.suptitle("Exp 1: RFF Convergence Rate", fontsize=12, y=1.02)
    plt.tight_layout()
    fig_save(fig, os.path.join(out_dir, "convergence_rate.png"))

    wide_slope = all_slopes[TEMP_WIDE]
    print(f"\n  PASS (τ=0.3 slope in [-0.7, -0.3]): "
          f"{-0.7 <= wide_slope <= -0.3}")
    return all_slopes


# ─────────────────────────────────────────────────────────────
# Experiment 2 — Error Budget Comparison
# ─────────────────────────────────────────────────────────────

def run_exp2(save_dir: str, quick: bool = False):
    """
    Verify Theorem 4.3: RFF bias ≤ minibatch variance for D ≥ N.

    Uses Laplace RFF throughout.  The population drift V_pop and minibatch
    drift V_mb both use the exact Laplace kernel.  The RFF drift uses
    Laplace RFF to approximate the same kernel, so the only extra error
    is the O(1/√D) RFF approximation.
    """
    print("\n=== Experiment 2: Error Budget Comparison ===")
    out_dir = os.path.join(save_dir, "exp2_error_budget")
    os.makedirs(out_dir, exist_ok=True)

    n_query     = 20 if quick else 100
    n_pop       = 2048 if quick else 8192
    n_seeds     = 10 if quick else 50
    N_values    = [64, 128, 256, 512] if quick else [64, 128, 256, 512, 1024]
    D_ratios    = [0.5, 1.0, 2.0, 4.0]

    device = torch.device("cpu")
    query  = sample_ring_mixture(n_query, seed=10).to(device)
    neg_q  = sample_ring_mixture(256, seed=11).to(device)

    # Population drift (large N ≈ true expectation)
    pos_pop = sample_ring_mixture(n_pop, seed=12).to(device)
    with torch.no_grad():
        V_pop = compute_drift_exact(query, pos_pop, temp=TEMP_WIDE,
                                    v_norm=False, neg=neg_q)

    results = {N: {} for N in N_values}
    csv_rows = []

    for N in N_values:
        # Measure minibatch variance (exact kernel, varying N)
        var_mb_vals = []
        for seed in range(n_seeds):
            pos_mb = sample_ring_mixture(N, seed=seed * 7 + N).to(device)
            with torch.no_grad():
                V_mb = compute_drift_exact(query, pos_mb, temp=TEMP_WIDE,
                                           v_norm=False, neg=neg_q)
            mse = (V_mb - V_pop).norm(dim=-1).pow(2).mean().item()
            var_mb_vals.append(mse)
        var_mb = float(np.mean(var_mb_vals))

        for ratio in D_ratios:
            D = max(8, int(N * ratio))
            mse_total_vals = []
            for seed in range(n_seeds):
                pos_mb = sample_ring_mixture(N, seed=seed * 7 + N).to(device)
                with torch.no_grad():
                    V_rff = compute_drift_rff(
                        query, pos_mb, D=D, bandwidth=TEMP_WIDE,
                        v_norm=False, neg=neg_q,
                        kernel="laplace", seed=seed * 13)
                mse = (V_rff - V_pop).norm(dim=-1).pow(2).mean().item()
                mse_total_vals.append(mse)
            mse_total = float(np.mean(mse_total_vals))
            ratio_val = mse_total / (var_mb + 1e-12)
            results[N][ratio] = ratio_val
            csv_rows.append([N, D, ratio, ratio_val, var_mb, mse_total])
            print(f"  N={N:4d}, D/N={ratio:.1f} (D={D:4d}): "
                  f"MSE/Var_mb = {ratio_val:.3f}")

    save_csv(os.path.join(out_dir, "error_budget.csv"),
             ["N", "D", "D_ratio", "mse_ratio", "var_mb", "mse_total"],
             csv_rows)

    # Plot 1: Line plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = cm.viridis(np.linspace(0, 0.8, len(N_values)))
    for color, N in zip(colors, N_values):
        ratios = [results[N][r] for r in D_ratios]
        ax.plot(D_ratios, ratios, "o-", color=color, label=f"N={N}")
    ax.axhline(y=1.0, color="gray", linestyle="--", label="MSE = Var_mb")
    ax.axhline(y=2.0, color="red",  linestyle=":",
               label="2× Var_mb (pass threshold)")
    ax.set_xlabel("D/N ratio")
    ax.set_ylabel("MSE_total / Var_mb")
    ax.set_title("Exp 2: Laplace RFF Error vs Mini-Batch Variance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig_save(fig, os.path.join(out_dir, "error_budget_lines.png"))

    # Plot 2: Heatmap
    ratio_matrix = np.array([[results[N][r] for r in D_ratios]
                               for N in N_values])
    fig, ax = plt.subplots(figsize=(5, 4))
    vmax = min(4.0, ratio_matrix.max())
    im = ax.imshow(ratio_matrix, aspect="auto", vmin=0, vmax=vmax,
                   cmap="RdYlGn_r")
    ax.set_xticks(range(len(D_ratios)))
    ax.set_xticklabels([str(r) for r in D_ratios])
    ax.set_yticks(range(len(N_values)))
    ax.set_yticklabels([str(N) for N in N_values])
    ax.set_xlabel("D/N ratio")
    ax.set_ylabel("Mini-batch size N")
    ax.set_title("Exp 2: MSE_total / Var_mb (Laplace RFF)")
    plt.colorbar(im, ax=ax, label="Ratio")
    for i in range(len(N_values)):
        for j in range(len(D_ratios)):
            ax.text(j, i, f"{ratio_matrix[i,j]:.2f}", ha="center",
                    va="center", fontsize=8, color="black")
    fig_save(fig, os.path.join(out_dir, "error_budget_heatmap.png"))

    return results


# ─────────────────────────────────────────────────────────────
# Experiment 3 — Direction Preservation
# ─────────────────────────────────────────────────────────────

def run_exp3(save_dir: str, quick: bool = False):
    """
    Cosine similarity between exact and RFF drift vectors.

    Uses Laplace RFF.  Also plots Gaussian RFF as negative control.
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
    query = sample_ring_mixture(n_query, seed=20).to(device)
    pos   = sample_ring_mixture(n_pos,   seed=21).to(device)
    neg   = sample_ring_mixture(n_neg,   seed=22).to(device)

    with torch.no_grad():
        V_exact = compute_drift_exact(query, pos, temp=TEMP_WIDE,
                                      v_norm=False, neg=neg)
        V_exact_n = V_exact / V_exact.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    all_results = {}

    for kernel_name in ["laplace", "gaussian"]:
        mean_cos, std_cos = [], []
        csv_rows = []

        for D in D_values:
            cos_seeds = []
            for seed in range(n_seeds):
                with torch.no_grad():
                    V_rff = compute_drift_rff(query, pos, D=D,
                                              bandwidth=TEMP_WIDE, v_norm=False,
                                              neg=neg, kernel=kernel_name,
                                              seed=seed)
                    V_rff_n = V_rff / V_rff.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    cos_sim = (V_exact_n * V_rff_n).sum(dim=-1).mean().item()
                cos_seeds.append(cos_sim)

            mu = float(np.mean(cos_seeds))
            sigma = float(np.std(cos_seeds))
            mean_cos.append(mu)
            std_cos.append(sigma)
            csv_rows.append([D, mu, sigma])
            print(f"  [{kernel_name:>8}] D={D:4d}: "
                  f"cosine = {mu:.4f} ± {sigma:.4f}")

        all_results[kernel_name] = (mean_cos, std_cos)
        save_csv(os.path.join(out_dir, f"cosine_{kernel_name}.csv"),
                 ["D", "mean_cosine", "std_cosine"], csv_rows)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"laplace": "steelblue", "gaussian": "tomato"}
    for kn in ["laplace", "gaussian"]:
        mu, se = all_results[kn]
        ax.errorbar(D_values, mu, yerr=se, fmt="o-", color=colors[kn],
                    capsize=4, label=f"RFF ({kn})")
    for y, label, style in [(0.90, "0.90 (pass)", "--"),
                             (0.95, "0.95",        ":"),
                             (0.99, "0.99",        "-.")]:
        ax.axhline(y=y, color="gray", linestyle=style, alpha=0.6, label=label)
    ax.set_xscale("log")
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlabel("RFF Dimension D")
    ax.set_ylabel("Mean Cosine Similarity cos(V, V_D)")
    ax.set_title("Exp 3: Drift Direction Preservation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig_save(fig, os.path.join(out_dir, "cosine_similarity.png"))

    d64_idx = D_values.index(64)
    lap_cos_64 = all_results["laplace"][0][d64_idx]
    print(f"\n  Laplace RFF cosine at D=64: {lap_cos_64:.4f}")
    print(f"  PASS (cosine > 0.9 at D=64): {lap_cos_64 > 0.9}")
    return all_results


# ─────────────────────────────────────────────────────────────
# Experiment 4 — Wall-Clock Speedup
# ─────────────────────────────────────────────────────────────

def run_exp4(save_dir: str, quick: bool = False):
    """
    Wall-clock speedup of Laplace RFF vs exact drift computation.

    Pre-samples W,b outside the timing loop so we measure only the
    kernel-approximation + drift computation, not the one-time sampling.
    """
    print("\n=== Experiment 4: Wall-Clock Speedup ===")
    out_dir = os.path.join(save_dir, "exp4_speedup")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cpu")

    d_values = [2, 64, 256] if quick else [2, 64, 256, 512, 1024]
    B_values = [128, 256, 512] if quick else [128, 256, 512, 1024]
    D_values = [64, 128] if quick else [64, 128, 256]
    n_warmup = 5  if quick else 20
    n_timed  = 10 if quick else 50

    csv_rows = []

    for d in d_values:
        for B in B_values:
            N = B
            P = B
            gen_data = torch.randn(B, d)
            pos_data = torch.randn(P, d)
            neg_data = torch.randn(N, d)

            # Time exact drift
            for _ in range(n_warmup):
                compute_drift_exact(gen_data, pos_data, temp=TEMP_WIDE,
                                    v_norm=False, neg=neg_data)
            times_exact = []
            for _ in range(n_timed):
                t0 = time.perf_counter()
                compute_drift_exact(gen_data, pos_data, temp=TEMP_WIDE,
                                    v_norm=False, neg=neg_data)
                times_exact.append(time.perf_counter() - t0)
            t_exact = float(np.median(times_exact)) * 1000  # ms

            for D in D_values:
                # Pre-sample W, b (amortized cost, done once per epoch)
                W, bb = sample_rff_params(d, D, TEMP_WIDE, device,
                                          kernel="laplace", seed=0)

                for _ in range(n_warmup):
                    compute_drift_rff(gen_data, pos_data, D=D,
                                      bandwidth=TEMP_WIDE, v_norm=False,
                                      neg=neg_data, kernel="laplace",
                                      W=W, b=bb)
                times_rff = []
                for _ in range(n_timed):
                    t0 = time.perf_counter()
                    compute_drift_rff(gen_data, pos_data, D=D,
                                      bandwidth=TEMP_WIDE, v_norm=False,
                                      neg=neg_data, kernel="laplace",
                                      W=W, b=bb)
                    times_rff.append(time.perf_counter() - t0)
                t_rff = float(np.median(times_rff)) * 1000

                speedup = t_exact / (t_rff + 1e-9)
                csv_rows.append([d, B, D, t_exact, t_rff, speedup])
                print(f"  d={d:4d}, B={B:4d}, D={D:3d}: "
                      f"exact={t_exact:.2f}ms, RFF={t_rff:.2f}ms, "
                      f"speedup={speedup:.2f}x")

    save_csv(os.path.join(out_dir, "speedup_results.csv"),
             ["d", "B", "D", "t_exact_ms", "t_rff_ms", "speedup"],
             csv_rows)

    # Plot: speedup vs B for each d at D = median entry
    D_plot = D_values[len(D_values) // 2]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    palette = cm.plasma(np.linspace(0.1, 0.9, len(d_values)))
    for color, d in zip(palette, d_values):
        speedups = [r[5] for r in csv_rows
                    if r[0] == d and r[2] == D_plot]
        Bs = [r[1] for r in csv_rows
              if r[0] == d and r[2] == D_plot]
        ax.plot(Bs, speedups, "o-", color=color, label=f"d={d}")
    ax.axhline(y=1.0, color="gray", linestyle="--", label="No speedup")
    ax.set_xlabel("Batch Size B (= N)")
    ax.set_ylabel(f"Speedup (exact / Laplace RFF) at D={D_plot}")
    ax.set_title("Exp 4: Wall-Clock Speedup")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig_save(fig, os.path.join(out_dir, "speedup_vs_batch.png"))

    # Heatmap: d × B for D_plot
    speedup_matrix = np.array(
        [[next((r[5] for r in csv_rows
                if r[0] == d and r[1] == B and r[2] == D_plot), 0)
          for B in B_values]
         for d in d_values]
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
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

def _train_one(method: str, D: int, kernel: str, n_steps: int,
               eval_every: int, seed: int, batch_size: int) -> dict:
    """Train one MLP generator and return loss + MMD curves."""
    device = torch.device("cpu")
    torch.manual_seed(seed)

    model = MLP(in_dim=32, hidden=256, out_dim=2).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    ref_gt = sample_ring_mixture(10000, seed=99).to(device)

    if method == "exact":
        drift_fn = partial(compute_drift_exact, temp=TEMP_WIDE, v_norm=True)
    elif method == "rff":
        drift_fn = partial(compute_drift_rff, D=D, bandwidth=TEMP_WIDE,
                           v_norm=True, kernel=kernel)
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
    End-to-end generation quality.

    Primary: Laplace RFF at D=16,32,64,128  (should match exact).
    Negative control: Gaussian RFF at D=128  (should fail).
    """
    print("\n=== Experiment 5: End-to-End Generation Quality ===")
    out_dir = os.path.join(save_dir, "exp5_quality")
    os.makedirs(out_dir, exist_ok=True)

    n_seeds    = 2    if quick else 5
    n_steps    = 500  if quick else 3000
    eval_every = 50   if quick else 300
    batch_size = 256  if quick else 1024
    D_values   = [32, 128] if quick else [16, 32, 64, 128]

    # Methods: (key, method, D, kernel)
    methods_config = [
        ("exact",           "exact", None, None),
    ] + [
        (f"laplace_D{D}",  "rff",   D,    "laplace")
        for D in D_values
    ] + [
        ("gaussian_D128",   "rff",   128,  "gaussian"),
    ]

    all_results = {}
    gt_samples = sample_ring_mixture(5000, seed=999)

    for key, method, D, kernel in methods_config:
        if method == "exact":
            label = "Exact (Laplace)"
        elif kernel == "laplace":
            label = f"Laplace RFF D={D}"
        else:
            label = f"Gaussian RFF D={D} (ctrl)"

        print(f"\n  Training: {label}  ({n_seeds} seeds)")
        seed_results = []
        for seed_i in range(n_seeds):
            print(f"    Seed {seed_i}...", end=" ", flush=True)
            t0 = time.time()
            res = _train_one(method, D, kernel, n_steps, eval_every,
                             seed=42 + seed_i, batch_size=batch_size)
            elapsed = time.time() - t0
            res["wall_time"] = elapsed
            seed_results.append(res)
            print(f"MMD={res['final_mmd']:.5f}, "
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
                              f"{np.mean(mmds):.5f} +/- {np.std(mmds):.5f}",
                              f"{np.mean(times):.1f}s"])
        print(f"  {key:<18}: "
              f"MMD = {np.mean(mmds):.5f} +/- {np.std(mmds):.5f}")
    save_csv(os.path.join(out_dir, "summary.csv"),
             ["method", "final_mmd", "wall_time"], summary_rows)

    # Pass checks
    exact_mmd = np.mean([r["final_mmd"] for r in all_results["exact"]])
    for D in D_values:
        key = f"laplace_D{D}"
        rff_mmd = np.mean([r["final_mmd"] for r in all_results[key]])
        passed = rff_mmd <= exact_mmd * 1.5
        print(f"  PASS Laplace D={D:3d} (within 50% of exact): {passed}  "
              f"({rff_mmd:.5f} vs {exact_mmd:.5f})")

    # ── Plot 1: MMD curves ──
    # Assign colors
    colors = {"exact": "black", "gaussian_D128": "tomato"}
    palette = cm.viridis(np.linspace(0.1, 0.85, len(D_values)))
    for D, c in zip(D_values, palette):
        colors[f"laplace_D{D}"] = c

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, results in all_results.items():
        steps = results[0]["mmd_steps"]
        mmd_arr = np.array([r["mmd_history"] for r in results])
        mu = mmd_arr.mean(axis=0)
        se = mmd_arr.std(axis=0)
        if key == "exact":
            label = "Exact (Laplace)"
            lw, ls = 2.5, "-"
        elif key.startswith("laplace"):
            label = key.replace("laplace_D", "Laplace RFF D=")
            lw, ls = 1.5, "--"
        else:
            label = "Gaussian RFF D=128 (ctrl)"
            lw, ls = 1.5, ":"
        ax.plot(steps, mu, color=colors[key], lw=lw, ls=ls, label=label)
        ax.fill_between(steps, mu - se, mu + se,
                        alpha=0.12, color=colors[key])
    ax.set_xlabel("Training Step")
    ax.set_ylabel("MMD² to Ground Truth")
    ax.set_title("Exp 5: Generation Quality (lower = better)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig_save(fig, os.path.join(out_dir, "mmd_curves.png"))

    # ── Plot 2: Final scatter grid ──
    n_methods = len(all_results) + 1
    ncols = 3
    nrows = math.ceil(n_methods / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.3, nrows * 3.3))
    axes = axes.flatten()

    ax = axes[0]
    gt_np = gt_samples.numpy()
    ax.scatter(gt_np[:, 0], gt_np[:, 1], s=2, alpha=0.3, c="black")
    ax.set_title("Ground Truth", fontsize=9)
    ax.set_aspect("equal"); ax.axis("off")

    for i, (key, results) in enumerate(all_results.items()):
        ax = axes[i + 1]
        gen_np = results[0]["final_gen"].numpy()
        c = colors.get(key, "gray")
        if key == "exact":
            label = "Exact (Laplace)"
        elif key.startswith("laplace"):
            label = key.replace("laplace_D", "Lap RFF D=")
        else:
            label = "Gauss RFF D=128"
        mmd_val = np.mean([r["final_mmd"] for r in results])
        ax.scatter(gen_np[:, 0], gen_np[:, 1], s=2, alpha=0.3, c=[c])
        ax.set_title(f"{label}\nMMD²={mmd_val:.4f}", fontsize=8)
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
    Visualize drift fields and energy dissipation.

    Trains one model with exact Laplace drift, then at each snapshot
    computes exact, Laplace RFF, and Gaussian RFF (ctrl) fields.
    """
    print("\n=== Experiment 6: Vector Field Visualization + Energy ===")
    out_dir = os.path.join(save_dir, "exp6_visualization")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cpu")
    torch.manual_seed(0)

    n_steps    = 500  if quick else 3000
    snap_steps = [0, 100, 300] if quick else [0, 500, 1000, 2000, 3000]
    D_lap_vals = [32, 128] if quick else [32, 128, 512]
    D_gau_ctrl = 128  # Gaussian negative control
    batch_size = 256  if quick else 1024
    grid_res   = 12   if quick else 20
    energy_interval = 10 if quick else 50

    lin = torch.linspace(-1.2, 1.2, grid_res)
    gx, gy = torch.meshgrid(lin, lin, indexing="ij")
    grid_pts = torch.stack([gx.flatten(), gy.flatten()], dim=1)

    # Train model with exact drift
    model = MLP(in_dim=32, hidden=256, out_dim=2).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    drift_fn = partial(compute_drift_exact, temp=TEMP_WIDE, v_norm=True)

    # row_keys for the visualization grid
    row_keys = (["exact"] +
                [f"lap_D{D}" for D in D_lap_vals] +
                [f"gau_D{D_gau_ctrl}"])
    row_labels = (["Exact (Laplace)"] +
                  [f"Laplace RFF D={D}" for D in D_lap_vals] +
                  [f"Gaussian RFF D={D_gau_ctrl} (ctrl)"])

    snapshots = {}
    energy_log = {"steps": []}
    for rk in row_keys:
        energy_log[rk] = []

    for step in range(n_steps + 1):
        # Collect snapshot
        if step in snap_steps:
            with torch.no_grad():
                gen_snap = model(torch.randn(1024, 32, device=device))
                pos_snap = sample_ring_mixture(1024,
                                               seed=step * 3 + 1).to(device)
                snap = {}
                snap["exact"] = compute_drift_exact(
                    grid_pts, pos_snap, temp=TEMP_WIDE, v_norm=False,
                    neg=gen_snap).numpy()
                for D in D_lap_vals:
                    snap[f"lap_D{D}"] = compute_drift_rff(
                        grid_pts, pos_snap, D=D, bandwidth=TEMP_WIDE,
                        v_norm=False, neg=gen_snap,
                        kernel="laplace", seed=42).numpy()
                snap[f"gau_D{D_gau_ctrl}"] = compute_drift_rff(
                    grid_pts, pos_snap, D=D_gau_ctrl, bandwidth=TEMP_WIDE,
                    v_norm=False, neg=gen_snap,
                    kernel="gaussian", seed=42).numpy()

                snapshots[step] = {
                    "snap": snap,
                    "gen": gen_snap.numpy(),
                    "pos": pos_snap.numpy(),
                }

        # Energy proxy
        if step % energy_interval == 0:
            with torch.no_grad():
                gen_e = model(torch.randn(512, 32, device=device))
                pos_e = sample_ring_mixture(512, seed=step + 100).to(device)
                energy_log["steps"].append(step)

                V_e = compute_drift_exact(gen_e, pos_e, temp=TEMP_WIDE,
                                          v_norm=False)
                energy_log["exact"].append(
                    V_e.norm(dim=-1).pow(2).mean().item())
                for D in D_lap_vals:
                    V_r = compute_drift_rff(gen_e, pos_e, D=D,
                                            bandwidth=TEMP_WIDE, v_norm=False,
                                            kernel="laplace", seed=D)
                    energy_log[f"lap_D{D}"].append(
                        V_r.norm(dim=-1).pow(2).mean().item())
                V_g = compute_drift_rff(gen_e, pos_e, D=D_gau_ctrl,
                                        bandwidth=TEMP_WIDE, v_norm=False,
                                        kernel="gaussian", seed=99)
                energy_log[f"gau_D{D_gau_ctrl}"].append(
                    V_g.norm(dim=-1).pow(2).mean().item())

        if step == 0:
            continue

        pos = sample_ring_mixture(batch_size, seed=step).to(device)
        z   = torch.randn(batch_size, 32, device=device)
        gen = model(z)
        loss = drifting_loss_fn(gen, pos, drift_fn)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # ── Plot 1: Vector field grid ──
    n_rows = len(row_keys)
    n_cols = len(snap_steps)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.8, n_rows * 2.8))
    if n_rows == 1: axes = axes[np.newaxis, :]
    if n_cols == 1: axes = axes[:, np.newaxis]

    vmax = None

    for col_i, step in enumerate(snap_steps):
        sd = snapshots[step]["snap"]
        gp = snapshots[step]["gen"]
        pp = snapshots[step]["pos"]

        for row_i, (rk, rl) in enumerate(zip(row_keys, row_labels)):
            ax = axes[row_i, col_i]
            V = sd[rk]
            mag = np.linalg.norm(V, axis=-1)
            if vmax is None:
                vmax = np.percentile(mag, 95) + 1e-8

            ax.scatter(pp[:200, 0], pp[:200, 1], s=3, alpha=0.25,
                       c="black", zorder=2)
            ax.scatter(gp[:200, 0], gp[:200, 1], s=3, alpha=0.25,
                       c="tab:orange", zorder=2)
            ax.quiver(grid_pts[:, 0].numpy(), grid_pts[:, 1].numpy(),
                      V[:, 0], V[:, 1], mag,
                      cmap="coolwarm", alpha=0.8, scale=None,
                      width=0.004, clim=(0, vmax), zorder=3)
            ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
            ax.set_aspect("equal"); ax.axis("off")
            if col_i == 0:
                ax.set_ylabel(rl, fontsize=8, rotation=90,
                              labelpad=5, va="center")
            if row_i == 0:
                ax.set_title(f"Step {step}", fontsize=9)

    plt.suptitle("Exp 6: Drift Vector Fields\n"
                 "(black=real, orange=generated)", fontsize=10, y=1.01)
    plt.tight_layout()
    fig_save(fig, os.path.join(out_dir, "vector_fields.png"), dpi=120)

    # ── Plot 2: Energy dissipation ──
    fig, ax = plt.subplots(figsize=(8, 4.5))
    steps_arr = energy_log["steps"]
    ax.plot(steps_arr, energy_log["exact"],
            color="black", lw=2.5, label="Exact (Laplace)")
    palette = cm.viridis(np.linspace(0.1, 0.85, len(D_lap_vals)))
    for color, D in zip(palette, D_lap_vals):
        ax.plot(steps_arr, energy_log[f"lap_D{D}"],
                color=color, lw=1.5, ls="--", label=f"Laplace RFF D={D}")
    ax.plot(steps_arr, energy_log[f"gau_D{D_gau_ctrl}"],
            color="tomato", lw=1.5, ls=":", label=f"Gaussian RFF D={D_gau_ctrl} (ctrl)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Energy Proxy E[||V(gen)||²]")
    ax.set_title("Exp 6: Energy Dissipation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig_save(fig, os.path.join(out_dir, "energy_dissipation.png"))

    # Save CSV
    csv_rows = []
    for i, step in enumerate(steps_arr):
        row = [step, energy_log["exact"][i]]
        for D in D_lap_vals:
            row.append(energy_log[f"lap_D{D}"][i])
        row.append(energy_log[f"gau_D{D_gau_ctrl}"][i])
        csv_rows.append(row)
    save_csv(os.path.join(out_dir, "energy_dissipation.csv"),
             ["step", "exact"] +
             [f"laplace_D{D}" for D in D_lap_vals] +
             [f"gaussian_D{D_gau_ctrl}"],
             csv_rows)

    return snapshots, energy_log


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RFF-Drift Toy Experiments (v2 — fixed kernel matching)")
    parser.add_argument("--exp", type=int, choices=[0, 1, 2, 3, 4, 5, 6],
                        help="Run a single experiment (0–6)")
    parser.add_argument("--all", action="store_true",
                        help="Run all experiments (0–6) sequentially")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer seeds/steps")
    parser.add_argument("--save-dir", type=str,
                        default="toy_results/rff_experiments",
                        help="Root directory for outputs")
    args = parser.parse_args()

    if args.exp is None and not args.all:
        parser.print_help()
        return

    save_dir = args.save_dir
    q = args.quick

    if q:
        print("[QUICK MODE] Reduced configs for testing.\n")

    exp_fns = {
        0: lambda: run_exp0(save_dir, q),
        1: lambda: run_exp1(save_dir, q),
        2: lambda: run_exp2(save_dir, q),
        3: lambda: run_exp3(save_dir, q),
        4: lambda: run_exp4(save_dir, q),
        5: lambda: run_exp5(save_dir, q),
        6: lambda: run_exp6(save_dir, q),
    }

    to_run = list(range(0, 7)) if args.all else [args.exp]

    t_start = time.time()
    for exp_num in to_run:
        exp_fns[exp_num]()

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"All done.  Total time: {total:.1f}s")
    print(f"Results saved to: {save_dir}/")


if __name__ == "__main__":
    main()
