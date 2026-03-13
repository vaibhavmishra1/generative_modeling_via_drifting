"""
Drifting Field Computation.

Implements the mean-shift drifting field V_{p,q} from the paper:
  "Generative Modeling via Drifting" (Deng et al., 2025)

The drifting field V_{p,q}(x) is defined as:
  V_{p,q}(x) = V_p^+(x) - V_q^-(x)

where:
  V_p^+(x) = (1/Z_p(x)) * E_{y+ ~ p}[k(x, y+)(y+ - x)]   (attraction to data)
  V_q^-(x) = (1/Z_q(x)) * E_{y- ~ q}[k(x, y-)(y- - x)]   (repulsion from generated)

With batch-normalized kernel K_B for practical estimation.

The kernel used is: k(x, y) = exp(-||x - y|| / τ)
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple


def compute_drift(
    gen: torch.Tensor,
    pos: torch.Tensor,
    temp: float = 0.05,
    v_norm: bool = True,
    neg: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the drifting field V_{p,q}(x) for generated samples.

    This implements the mean-shift drifting field with batch-normalized kernels,
    as described in the paper. The field drives generated samples toward data
    samples (attraction) and away from other generated samples (repulsion).

    Args:
        gen: Generated samples [G, D] — these are the query points x
        pos: Positive (data) samples [P, D] — y+ drawn from p_data
        temp: Temperature τ for the kernel k(x,y) = exp(-||x-y||/τ)
        v_norm: Whether to normalize V to unit norm per sample (V-normalization)
        neg: Optional negative (generated) samples [N, D] from queue.
             If None, uses `gen` itself as negative samples.

    Returns:
        V: Drift vectors [G, D]
    """
    if neg is None:
        neg = gen  # Use generated samples as negatives

    G = gen.shape[0]
    N = neg.shape[0]

    # Concatenate targets: [neg; pos] with shape [N+P, D]
    targets = torch.cat([neg, pos], dim=0)

    # Compute pairwise distances: [G, N+P]
    dist = torch.cdist(gen, targets)

    # Mask self-distances if neg == gen (avoid trivial self-matching)
    if neg is gen:
        dist[:, :G].fill_diagonal_(1e6)

    # Kernel: k(x, y) = exp(-||x - y|| / τ)
    kernel = (-dist / temp).exp()

    # Batch-normalized kernel: normalize along both dimensions
    # K_B(x, y) = k(x, y) / sqrt(Z_x * Z_y)
    # where Z_x = sum_y k(x, y) and Z_y = sum_x k(x, y)
    row_sum = kernel.sum(dim=-1, keepdim=True)  # [G, 1]
    col_sum = kernel.sum(dim=-2, keepdim=True)  # [1, N+P]
    normalizer = (row_sum * col_sum).clamp_min(1e-12).sqrt()
    normalized_kernel = kernel / normalizer

    # Split kernel into negative and positive parts
    neg_kernel = normalized_kernel[:, :N]   # [G, N]
    pos_kernel = normalized_kernel[:, N:]   # [G, P]

    # Compute V+ (attraction toward data):
    # V_p^+(x) = sum_y+ K(x, y+) * sum_y- K(x, y-) * y+
    # (factored form from batch normalization)
    pos_weight = pos_kernel * neg_kernel.sum(dim=-1, keepdim=True)  # [G, P]
    V_pos = pos_weight @ targets[N:]  # [G, D]

    # Compute V- (repulsion from generated):
    neg_weight = neg_kernel * pos_kernel.sum(dim=-1, keepdim=True)  # [G, N]
    V_neg = neg_weight @ targets[:N]  # [G, D]

    V = V_pos - V_neg

    # V-normalization: normalize V to unit norm per sample
    # This stabilizes training by controlling the drift step size
    if v_norm:
        V_norms = V.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        V = V / V_norms

    return V


def compute_drift_multiscale(
    gen_features: List[torch.Tensor],
    pos_features: List[torch.Tensor],
    temps: List[float],
    v_norm: bool = True,
    neg_features: Optional[List[torch.Tensor]] = None,
    weights: Optional[List[float]] = None,
) -> List[torch.Tensor]:
    """
    Compute multi-scale drifting fields across multiple feature levels.

    The paper extracts features from multiple layers of a pretrained encoder
    and applies the drifting loss at each scale independently.

    Args:
        gen_features: List of generated features at different scales,
                      each [G, D_i] where D_i is the feature dim at scale i
        pos_features: List of data features at different scales
        temps: List of temperatures for each scale
        v_norm: Whether to apply V-normalization
        neg_features: Optional list of negative features from queue
        weights: Optional per-scale loss weights

    Returns:
        List of drift vectors, one per scale, each [G, D_i]
    """
    num_scales = len(gen_features)
    assert len(pos_features) == num_scales
    assert len(temps) == num_scales

    if neg_features is None:
        neg_features = [None] * num_scales
    if weights is None:
        weights = [1.0] * num_scales

    drifts = []
    for i in range(num_scales):
        V = compute_drift(
            gen=gen_features[i],
            pos=pos_features[i],
            temp=temps[i],
            v_norm=v_norm,
            neg=neg_features[i],
        )
        drifts.append(V * weights[i])

    return drifts
