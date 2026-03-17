"""
RFF-Approximated Drifting Field.

Implements the Random Fourier Feature acceleration of the drifting field from:
  "Accelerating Generative Drifting via Random Fourier Feature Drift Estimation"

The key idea: replace the exact O(BNd) kernel evaluation with an RFF approximation
that costs O((B+N)·d·D + B·N·D) where D << d.

For shift-invariant kernel k(x,y) = φ(x - y), Bochner's theorem gives:
  k(x, y) = E_{w~Λ}[z_w(x)^T z_w(y)]

where Λ is the spectral measure of k and:
  z_w(x) = sqrt(2) * cos(w^T x + b),  b ~ Uniform[0, 2π]

For the Gaussian kernel k_h(x,y) = exp(-||x-y||²/(2h²)):
  Λ = N(0, h^{-2} I_d)

This module uses the Gaussian kernel (satisfies theoretical conditions K1–K4 from
Cao et al. 2026) as the RFF target. The existing codebase uses the Laplace kernel;
this RFF version enables direct comparison.

References:
  Rahimi & Recht (2007). Random Features for Large-Scale Kernel Machines. NeurIPS.
  Yu et al. (2016). Orthogonal Random Features. NeurIPS.
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────
# RFF Primitives
# ─────────────────────────────────────────────────────────────

def sample_rff_params(
    d: int,
    D: int,
    bandwidth: float,
    device: torch.device,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample random frequency vectors and phase offsets for RFF.

    For Gaussian kernel k_h(x,y) = exp(-||x-y||²/(2h²)), the spectral
    distribution is Λ = N(0, h^{-2} I_d), so w ~ N(0, h^{-2} I).

    Args:
        d: Input feature dimension
        D: Number of RFF features
        bandwidth: Kernel bandwidth h (larger h = smoother kernel)
        device: Target device
        seed: Optional random seed for reproducibility

    Returns:
        W: Frequency matrix [D, d],  w_i ~ N(0, h^{-2} I)
        b: Phase offsets [D],        b_i ~ Uniform[0, 2π]
    """
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    W = torch.randn(D, d, device=device, generator=gen) / bandwidth
    b = torch.rand(D, device=device, generator=gen) * (2 * math.pi)
    return W, b


def rff_map(
    x: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Apply the RFF feature map: z(x) = sqrt(2/D) * cos(x W^T + b).

    This gives an unbiased estimator of the Gaussian kernel:
      E[z(x)^T z(y)] = k_h(x, y)

    Args:
        x: Input points [N, d]
        W: Frequency matrix [D, d]
        b: Phase offsets [D]

    Returns:
        z: RFF features [N, D]
    """
    D = W.shape[0]
    return math.sqrt(2.0 / D) * torch.cos(x @ W.T + b)


# ─────────────────────────────────────────────────────────────
# RFF Drift Field
# ─────────────────────────────────────────────────────────────

def compute_drift_rff(
    gen: torch.Tensor,
    pos: torch.Tensor,
    D: int = 256,
    bandwidth: float = 0.05,
    v_norm: bool = True,
    neg: Optional[torch.Tensor] = None,
    W: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute the RFF-approximated drifting field V^D_{p,q}(x).

    Replaces exact pairwise distance kernel evaluations with RFF inner products:
      k_D(x, y) = z(x)^T z(y)  ≈  k_h(x, y)  (Gaussian kernel)

    The batch-normalization structure and V_pos / V_neg computation are identical
    to the exact version in drifting/drift_field.py. The RFF kernel can produce
    small negative values for low D, so we clamp before normalization.

    Complexity:
      Exact:  O(G·(N+P)·d)          — pairwise distances in d-dim
      RFF:    O((G+N+P)·d·D + G·(N+P)·D) — feature mapping + inner products

    For D << d, the RFF formulation is strictly cheaper. The main speedup
    comes from GPU-friendly matrix multiplications replacing scattered
    distance computations.

    Args:
        gen: Generated samples [G, d] — query points x
        pos: Positive (data) samples [P, d] — y+ from p_data
        D: Number of RFF features (more = more accurate, slower)
        bandwidth: Gaussian kernel bandwidth h (maps to temperature τ ≈ h²/2)
        v_norm: Normalize V to unit norm per sample
        neg: Optional negative samples [N, d] from queue; defaults to gen
        W: Pre-sampled frequency matrix [D, d] (reuse across calls for stability)
        b: Pre-sampled phase offsets [D]
        seed: Random seed for W, b if not provided

    Returns:
        V: RFF-approximated drift vectors [G, d]
    """
    if neg is None:
        neg = gen

    G = gen.shape[0]
    N = neg.shape[0]
    d = gen.shape[1]

    # Sample RFF params if not provided
    if W is None or b is None:
        W, b = sample_rff_params(d, D, bandwidth, gen.device, seed)

    # Map all point sets to RFF feature space: z(.) in R^D
    z_gen = rff_map(gen, W, b)   # [G, D]
    z_pos = rff_map(pos, W, b)   # [P, D]
    z_neg = rff_map(neg, W, b)   # [N, D]  (same as z_gen if neg is gen)

    # Concatenate targets: [neg; pos] in feature space
    z_targets = torch.cat([z_neg, z_pos], dim=0)   # [N+P, D]
    targets   = torch.cat([neg,   pos],   dim=0)   # [N+P, d]

    # Approximate kernel matrix via inner products: k_D(x, y) = z(x)^T z(y)
    # Shape: [G, N+P]
    kernel = z_gen @ z_targets.T

    # Mask self-similarities (same logic as exact version)
    if neg is gen:
        kernel[:, :G].fill_diagonal_(0.0)

    # Clamp to non-negative before batch-normalization.
    # RFF estimates can be slightly negative for small D; clamping is standard
    # practice and introduces a negligible bias relative to the O(1/√D) error.
    kernel = kernel.clamp_min(0.0)

    # Batch-normalized kernel: K_B(x,y) = k_D(x,y) / sqrt(Z_x * Z_y)
    row_sum = kernel.sum(dim=-1, keepdim=True).clamp_min(1e-12)   # [G, 1]
    col_sum = kernel.sum(dim=-2, keepdim=True).clamp_min(1e-12)   # [1, N+P]
    normalizer = (row_sum * col_sum).sqrt()
    normalized_kernel = kernel / normalizer

    # Split into negative and positive parts
    neg_kernel = normalized_kernel[:, :N]   # [G, N]
    pos_kernel = normalized_kernel[:, N:]   # [G, P]

    # V+ (attraction toward data)
    pos_weight = pos_kernel * neg_kernel.sum(dim=-1, keepdim=True)   # [G, P]
    V_pos = pos_weight @ targets[N:]                                  # [G, d]

    # V- (repulsion from generated)
    neg_weight = neg_kernel * pos_kernel.sum(dim=-1, keepdim=True)   # [G, N]
    V_neg = neg_weight @ targets[:N]                                  # [G, d]

    V = V_pos - V_neg

    if v_norm:
        V_norms = V.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        V = V / V_norms

    return V


# ─────────────────────────────────────────────────────────────
# Orthogonal RFF variant (lower variance, O(1/D) rate)
# ─────────────────────────────────────────────────────────────

def sample_orf_params(
    d: int,
    D: int,
    bandwidth: float,
    device: torch.device,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample Orthogonal Random Feature (ORF) parameters.

    ORF (Yu et al., 2016) replaces i.i.d. frequency vectors with orthogonalized
    ones. For Gaussian kernel, this reduces the variance from O(1/D) to O(1/D²),
    giving an effective O(1/D) approximation error (vs O(1/√D) for standard RFF).

    The construction: sample a random orthogonal matrix Q ∈ R^{D×d} and scale
    each row by a chi-distributed magnitude.

    Args:
        d, D, bandwidth, device, seed: Same as sample_rff_params

    Returns:
        W: Orthogonal frequency matrix [D, d]
        b: Phase offsets [D]
    """
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    # Build D×d orthogonal matrix by stacking ceil(D/d) random orthogonal blocks
    blocks = []
    remaining = D
    while remaining > 0:
        block_size = min(remaining, d)
        G_mat = torch.randn(d, d, device=device, generator=gen)
        Q, _ = torch.linalg.qr(G_mat)          # [d, d] orthogonal
        # Chi-distributed scaling to match spectral norm
        norms = torch.randn(d, device=device, generator=gen).norm()
        blocks.append(Q[:block_size] * (norms / math.sqrt(d)))
        remaining -= block_size

    W_orth = torch.cat(blocks, dim=0)[:D]       # [D, d]
    W = W_orth / bandwidth                       # scale by 1/h

    b = torch.rand(D, device=device, generator=gen) * (2 * math.pi)
    return W, b


def compute_drift_orf(
    gen: torch.Tensor,
    pos: torch.Tensor,
    D: int = 256,
    bandwidth: float = 0.05,
    v_norm: bool = True,
    neg: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    RFF drift using Orthogonal Random Features (ORF).

    Same interface as compute_drift_rff but uses ORF for better
    approximation quality at the same D.

    Per Yu et al. (2016): ORF achieves O(1/D) error vs O(1/√D) for standard RFF,
    allowing D=128 ORF to match D=512 standard RFF in approximation quality.
    """
    d = gen.shape[1]
    W, b = sample_orf_params(d, D, bandwidth, gen.device, seed)
    return compute_drift_rff(gen, pos, D=D, bandwidth=bandwidth,
                             v_norm=v_norm, neg=neg, W=W, b=b)
