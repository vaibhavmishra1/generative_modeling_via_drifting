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

Supported kernels:

  **Laplace** (default — matches the exact drift in drift_field.py):
    k(x, y) = exp(-||x - y|| / τ)
    Spectral measure Λ = multivariate Cauchy(0, 1/τ · I_d)
    Sampling: ω = z / (τ √s),  z ~ N(0, I_d),  s ~ χ²(1)

  **Gaussian** (theoretically preferred, satisfies K1–K4):
    k(x, y) = exp(-||x - y||² / (2h²))
    Spectral measure Λ = N(0, h⁻² I_d)

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
    kernel: str = "laplace",
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample random frequency vectors and phase offsets for RFF.

    The spectral distribution depends on the target kernel:

    **Laplace** k(x,y) = exp(-||x-y||/τ):
      The d-dimensional Fourier transform of the Laplace kernel is
        k̂(ω) ∝ 1 / (1 + τ²||ω||²)^{(d+1)/2}
      which is the density of a d-variate Cauchy(0, 1/τ · I).
      Sampling: ω = z / (τ √s)  where z ~ N(0, I_d), s ~ χ²(1).

    **Gaussian** k(x,y) = exp(-||x-y||²/(2h²)):
      Spectral density is N(0, h⁻² I_d).
      Sampling: ω ~ N(0, h⁻² I_d).

    Args:
        d: Input feature dimension
        D: Number of RFF features
        bandwidth: Kernel parameter (τ for Laplace, h for Gaussian)
        device: Target device
        kernel: "laplace" or "gaussian"
        seed: Optional random seed for reproducibility

    Returns:
        W: Frequency matrix [D, d]
        b: Phase offsets [D], b_i ~ Uniform[0, 2π]
    """
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    if kernel == "laplace":
        # Multivariate Cauchy: ω = z / (τ √s)
        # where z ~ N(0, I_d) and s ~ χ²(1) = N(0,1)²
        z = torch.randn(D, d, device=device, generator=gen)
        # s_i ~ χ²(1) for each frequency vector (scalar per row)
        s = torch.randn(D, 1, device=device, generator=gen).pow(2)
        # Clamp to avoid division by zero (limits max frequency)
        s = s.clamp_min(1e-8)
        W = z / (bandwidth * s.sqrt())
    elif kernel == "gaussian":
        W = torch.randn(D, d, device=device, generator=gen) / bandwidth
    else:
        raise ValueError(f"Unknown kernel: {kernel!r}. Use 'laplace' or 'gaussian'.")

    b = torch.rand(D, device=device, generator=gen) * (2 * math.pi)
    return W, b


def rff_map(
    x: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Apply the RFF feature map: z(x) = sqrt(2/D) * cos(x W^T + b).

    This gives an unbiased estimator of the target kernel:
      E[z(x)^T z(y)] = k(x, y)

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
    kernel: str = "laplace",
    W: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute the RFF-approximated drifting field V^D_{p,q}(x).

    Replaces exact pairwise-distance kernel evaluations with RFF inner
    products.  The batch-normalization, V_pos / V_neg split, and
    v-normalization are identical to the exact version in drift_field.py.

    When ``kernel="laplace"`` (the default), this approximates the *same*
    Laplace kernel used by ``compute_drift()`` in drift_field.py, giving a
    true drop-in replacement whose approximation error converges at
    O(1/√D).

    Complexity:
      Exact:  O(G·(N+P)·d)               — pairwise distances in d-dim
      RFF:    O((G+N+P)·d·D + G·(N+P)·D) — feature mapping + inner products

    Args:
        gen: Generated samples [G, d] — query points x
        pos: Positive (data) samples [P, d] — y+ from p_data
        D: Number of RFF features (more = more accurate, slower)
        bandwidth: Kernel parameter (τ for Laplace, h for Gaussian)
        v_norm: Normalize V to unit norm per sample
        neg: Optional negative samples [N, d] from queue; defaults to gen
        kernel: "laplace" (default, matches exact code) or "gaussian"
        W: Pre-sampled frequency matrix [D, d] (reuse for stability)
        b: Pre-sampled phase offsets [D]
        seed: Random seed for W, b if not provided

    Returns:
        V: RFF-approximated drift vectors [G, d]
    """
    neg_is_gen = neg is None
    if neg is None:
        neg = gen

    G = gen.shape[0]
    N = neg.shape[0]
    d = gen.shape[1]

    # Sample RFF params if not provided
    if W is None or b is None:
        W, b = sample_rff_params(d, D, bandwidth, gen.device,
                                 kernel=kernel, seed=seed)

    # Map all point sets to RFF feature space: z(.) in R^D
    z_gen = rff_map(gen, W, b)   # [G, D]
    z_pos = rff_map(pos, W, b)   # [P, D]
    if neg_is_gen:
        z_neg = z_gen
    else:
        z_neg = rff_map(neg, W, b)  # [N, D]

    # Concatenate targets: [neg; pos] in feature space
    z_targets = torch.cat([z_neg, z_pos], dim=0)   # [N+P, D]
    targets   = torch.cat([neg,   pos],   dim=0)   # [N+P, d]

    # Approximate kernel matrix via inner products: k_D(x, y) = z(x)^T z(y)
    # Shape: [G, N+P]
    kernel_mat = z_gen @ z_targets.T

    # Mask self-similarities (same logic as exact version)
    if neg_is_gen:
        kernel_mat[:, :G].fill_diagonal_(0.0)

    # ── Noise-aware thresholding ──
    # RFF inner products for distant pairs (true kernel ≈ 0) have std ≈ √(2/D).
    # After naive clamp-to-≥0, this creates a positive noise floor ≈ 0.8·√(2/D)
    # that corrupts the batch normalization (inflates row_sum / col_sum).
    #
    # Fix: subtract a fraction of the noise std before clamping. This removes
    # the noise floor while preserving signal for pairs with k(x,y) >> noise.
    # The threshold is conservative (0.5σ) to avoid discarding real signal.
    noise_std = math.sqrt(2.0 / D)
    kernel_mat = (kernel_mat - 0.5 * noise_std).clamp_min(0.0)

    # Batch-normalized kernel: K_B(x,y) = k_D(x,y) / sqrt(Z_x * Z_y)
    row_sum = kernel_mat.sum(dim=-1, keepdim=True).clamp_min(1e-12)   # [G, 1]
    col_sum = kernel_mat.sum(dim=-2, keepdim=True).clamp_min(1e-12)   # [1, N+P]
    normalizer = (row_sum * col_sum).sqrt()
    normalized_kernel = kernel_mat / normalizer

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
# Orthogonal RFF variant (lower variance, Gaussian kernel only)
# ─────────────────────────────────────────────────────────────

def sample_orf_params(
    d: int,
    D: int,
    bandwidth: float,
    device: torch.device,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample Orthogonal Random Feature (ORF) parameters (Gaussian kernel).

    ORF (Yu et al., 2016) replaces i.i.d. frequency vectors with
    orthogonalized ones, reducing variance from O(1/D) to O(1/D²)
    and effective approximation error from O(1/√D) to O(1/D).

    Note: ORF is only well-defined for the Gaussian kernel because the
    orthogonalization relies on rotational invariance of N(0, I).  For the
    Laplace kernel (Cauchy spectral distribution), standard i.i.d. RFF is
    used instead.

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
        # Chi-distributed row norms to match the Gaussian spectral magnitude
        norms = torch.randn(d, d, device=device, generator=gen).norm(dim=-1)
        blocks.append(Q[:block_size] * norms[:block_size].unsqueeze(-1))
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
    RFF drift using Orthogonal Random Features (Gaussian kernel only).

    Same interface as compute_drift_rff but uses ORF for better
    approximation quality at the same D.

    Per Yu et al. (2016): ORF achieves O(1/D) error vs O(1/√D) for
    standard RFF, allowing D=128 ORF to match D=512 standard RFF.
    """
    d = gen.shape[1]
    W, b = sample_orf_params(d, D, bandwidth, gen.device, seed)
    return compute_drift_rff(gen, pos, D=D, bandwidth=bandwidth,
                             v_norm=v_norm, neg=neg, kernel="gaussian",
                             W=W, b=b)
