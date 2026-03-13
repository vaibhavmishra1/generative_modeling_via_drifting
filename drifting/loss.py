"""
Drifting Loss.

Implements the training objective from the paper:
  L = E_ε[||φ(f_θ(ε)) - stopgrad(φ(f_θ(ε)) + V_{p,q}(φ(f_θ(ε))))||²]

Which simplifies to:
  L = E_ε[||V_{p,q}(φ(f_θ(ε)))||²]

The gradient flows through the encoder φ into the generator f_θ,
while the drift computation (V and the target) are detached.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from .drift_field import compute_drift, compute_drift_multiscale


def drifting_loss(
    gen: torch.Tensor,
    pos: torch.Tensor,
    temp: float = 0.05,
    v_norm: bool = True,
    neg: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the drifting loss for a single scale.

    L = ||gen - stopgrad(gen + V(gen, pos))||²
      = ||V(gen, pos)||²  (since stopgrad makes V a constant)

    But the key insight is that `gen` carries gradients from the generator,
    while the target `gen + V` is fully detached. This means the generator
    learns to produce samples that are already at the drifted position.

    Args:
        gen: Generated samples [G, D] (carries gradient)
        pos: Data samples [P, D] (detached, from data)
        temp: Kernel temperature
        v_norm: Whether to V-normalize
        neg: Optional negative samples from queue

    Returns:
        Scalar loss value
    """
    with torch.no_grad():
        V = compute_drift(gen, pos, temp=temp, v_norm=v_norm, neg=neg)
        target = (gen + V).detach()

    return F.mse_loss(gen, target)


class DriftingLoss(nn.Module):
    """
    Multi-scale drifting loss module.

    Computes the drifting loss across multiple feature scales from
    a pretrained encoder. Each scale has its own temperature and weight.

    Args:
        temps: List of temperatures for each feature scale
        weights: List of loss weights for each scale
        v_norm: Whether to apply V-normalization
    """

    def __init__(
        self,
        temps: List[float] = [0.05, 0.05, 0.05],
        weights: List[float] = [1.0, 1.0, 1.0],
        v_norm: bool = True,
    ):
        super().__init__()
        self.temps = temps
        self.weights = weights
        self.v_norm = v_norm
        assert len(temps) == len(weights)

    def forward(
        self,
        gen_features: List[torch.Tensor],
        pos_features: List[torch.Tensor],
        neg_features: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute total multi-scale drifting loss.

        Args:
            gen_features: List of [G, D_i] tensors from encoding generated samples
            pos_features: List of [P, D_i] tensors from encoding data samples
            neg_features: Optional list of [N, D_i] from queue

        Returns:
            Total scalar loss
        """
        num_scales = len(gen_features)
        assert num_scales == len(self.temps), (
            f"Expected {len(self.temps)} scales, got {num_scales}"
        )

        if neg_features is None:
            neg_features = [None] * num_scales

        total_loss = 0.0
        for i in range(num_scales):
            gen_feat = gen_features[i]  # [G, D_i] — carries gradient
            pos_feat = pos_features[i].detach()  # [P, D_i]
            neg_feat = neg_features[i].detach() if neg_features[i] is not None else None

            with torch.no_grad():
                V = compute_drift(
                    gen=gen_feat.detach(),  # Detach for drift computation
                    pos=pos_feat,
                    temp=self.temps[i],
                    v_norm=self.v_norm,
                    neg=neg_feat,
                )
                target = (gen_feat.detach() + V).detach()

            # Loss: the gen_feat here carries gradient through encoder → generator
            scale_loss = F.mse_loss(gen_feat, target)
            total_loss = total_loss + self.weights[i] * scale_loss

        return total_loss
