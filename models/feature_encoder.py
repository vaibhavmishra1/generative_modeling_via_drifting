"""
Feature Encoder for Drifting Models.

Provides multi-scale feature extraction using pretrained self-supervised
models (MAE, MoCo v3, etc.) for computing the drifting field in
feature space rather than pixel/latent space.

From the paper:
- Uses pretrained ViT-L/16 (MAE) as the primary feature encoder
- Extracts features from multiple intermediate blocks for multi-scale loss
- Features are L2-normalized before drift computation
- The encoder is frozen during training (no gradient updates)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from functools import partial

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class FeatureEncoder(nn.Module):
    """
    Base class for multi-scale feature extraction.

    Wraps a pretrained vision encoder and extracts features from
    specified intermediate layers.
    """

    def __init__(self):
        super().__init__()
        self.feature_dims: List[int] = []
        self.num_scales: int = 0

    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features from input images.

        Args:
            x: Input images [B, C, H, W] in range [0, 1] or [-1, 1]

        Returns:
            List of feature tensors [B, D_i], one per scale
        """
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract features with no gradient tracking."""
        return self.extract_features(x)


class MAEFeatureEncoder(FeatureEncoder):
    """
    Multi-scale feature encoder using MAE (Masked Autoencoder) pretrained ViT.

    Uses a ViT-L/16 pretrained with MAE on ImageNet. Extracts features from
    multiple transformer blocks and spatially pools them to get per-image
    feature vectors.

    The paper uses features from blocks at approximately 1/3, 2/3, and the
    final layer of the encoder for multi-scale representation.

    Args:
        model_name: timm model name for the pretrained MAE ViT
        extract_blocks: Which transformer block indices to extract features from
                       Default [7, 15, 23] for ViT-L (24 blocks)
        normalize: Whether to L2-normalize features
        input_size: Expected input image size
    """

    def __init__(
        self,
        model_name: str = "vit_large_patch16_224.mae",
        extract_blocks: List[int] = [7, 15, 23],
        normalize: bool = True,
        input_size: int = 256,
    ):
        super().__init__()
        assert HAS_TIMM, "timm is required for MAEFeatureEncoder. Install with: pip install timm"

        self.normalize = normalize
        self.extract_blocks = sorted(extract_blocks)

        # Load pretrained MAE ViT-L/16
        self.encoder = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # Remove classification head
            img_size=input_size,
        )

        # Freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        # Determine feature dimensions
        # ViT-L has hidden_dim=1024
        hidden_dim = self.encoder.embed_dim
        self.feature_dims = [hidden_dim] * len(extract_blocks)
        self.num_scales = len(extract_blocks)

        # Register hooks to capture intermediate features
        self._features = {}
        self._register_hooks()

        # ImageNet normalization
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _register_hooks(self):
        """Register forward hooks on specified blocks."""
        for block_idx in self.extract_blocks:
            block = self.encoder.blocks[block_idx]
            block.register_forward_hook(
                self._make_hook(block_idx)
            )

    def _make_hook(self, block_idx: int):
        def hook(module, input, output):
            self._features[block_idx] = output
        return hook

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input images for the pretrained encoder."""
        # Assume input is in [-1, 1], convert to [0, 1] first
        if x.min() < 0:
            x = (x + 1) / 2
        # Apply ImageNet normalization
        x = (x - self.mean) / self.std
        return x

    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features from images.

        Args:
            x: [B, C, H, W] images (in [-1, 1] or [0, 1] range)

        Returns:
            List of [B, D] feature vectors, one per extract_block
        """
        self._features.clear()

        # Normalize input
        x = self._normalize_input(x)

        # Resize if needed (MAE typically uses 224x224)
        if x.shape[-1] != self.encoder.patch_embed.img_size[0]:
            x = F.interpolate(
                x,
                size=(self.encoder.patch_embed.img_size[0],
                      self.encoder.patch_embed.img_size[1]),
                mode="bilinear",
                align_corners=False,
            )

        # Forward pass through encoder (hooks capture intermediate features)
        _ = self.encoder(x)

        # Collect and process features
        features = []
        for block_idx in self.extract_blocks:
            feat = self._features[block_idx]  # [B, N+1, D] or [B, N, D]

            # Global average pooling over spatial tokens (skip CLS if present)
            if hasattr(self.encoder, 'cls_token') and self.encoder.cls_token is not None:
                # Skip CLS token, pool spatial tokens
                feat = feat[:, 1:, :].mean(dim=1)  # [B, D]
            else:
                feat = feat.mean(dim=1)  # [B, D]

            # L2 normalize
            if self.normalize:
                feat = F.normalize(feat, dim=-1)

            features.append(feat)

        self._features.clear()
        return features

    def train(self, mode: bool = True):
        """Override to keep encoder in eval mode always."""
        super().train(mode)
        self.encoder.eval()
        return self


class DINOv2FeatureEncoder(FeatureEncoder):
    """
    Alternative feature encoder using DINOv2 (for ablation studies).

    DINOv2 provides strong self-supervised features that may work
    as alternatives to MAE features.

    Args:
        model_name: DINOv2 model variant
        extract_blocks: Block indices to extract from
        normalize: Whether to L2-normalize features
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitl14",
        extract_blocks: List[int] = [7, 15, 23],
        normalize: bool = True,
    ):
        super().__init__()
        self.normalize = normalize
        self.extract_blocks = sorted(extract_blocks)

        # Load DINOv2
        self.encoder = torch.hub.load("facebookresearch/dinov2", model_name)

        # Freeze
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        hidden_dim = self.encoder.embed_dim
        self.feature_dims = [hidden_dim] * len(extract_blocks)
        self.num_scales = len(extract_blocks)

        # Hooks
        self._features = {}
        for block_idx in self.extract_blocks:
            self.encoder.blocks[block_idx].register_forward_hook(
                self._make_hook(block_idx)
            )

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _make_hook(self, block_idx):
        def hook(module, input, output):
            self._features[block_idx] = output
        return hook

    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        self._features.clear()
        if x.min() < 0:
            x = (x + 1) / 2
        x = (x - self.mean) / self.std

        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        _ = self.encoder(x)

        features = []
        for block_idx in self.extract_blocks:
            feat = self._features[block_idx]
            if hasattr(feat, 'shape') and feat.dim() == 3:
                feat = feat[:, 1:, :].mean(dim=1) if feat.shape[1] > 1 else feat.squeeze(1)
            if self.normalize:
                feat = F.normalize(feat, dim=-1)
            features.append(feat)

        self._features.clear()
        return features

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self


def build_feature_encoder(
    encoder_type: str = "mae",
    extract_blocks: List[int] = [7, 15, 23],
    input_size: int = 256,
    **kwargs,
) -> FeatureEncoder:
    """
    Factory function to build a feature encoder.

    Args:
        encoder_type: Type of encoder ("mae", "dinov2")
        extract_blocks: Block indices to extract features from
        input_size: Input image size
    """
    if encoder_type == "mae":
        return MAEFeatureEncoder(
            extract_blocks=extract_blocks,
            input_size=input_size,
            **kwargs,
        )
    elif encoder_type == "dinov2":
        return DINOv2FeatureEncoder(
            extract_blocks=extract_blocks,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
