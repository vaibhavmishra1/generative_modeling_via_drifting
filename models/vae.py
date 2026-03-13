"""
VAE Wrapper for Drifting Models.

Wraps the Stable Diffusion VAE (AutoencoderKL from diffusers) for:
1. Encoding real images to latent space during training
2. Decoding generated latents to pixel space during inference

The latent space model uses the SD-VAE with:
- Latent channels: 4
- Spatial downsampling: 8x (256x256 → 32x32)
- Scaling factor: 0.18215 (SD 1.x) or 0.13025 (SDXL)

The VAE is frozen during training.
"""

import torch
import torch.nn as nn
from typing import Optional


class AutoencoderKLWrapper(nn.Module):
    """
    Wrapper around diffusers AutoencoderKL for the Drifting Model pipeline.

    Handles:
    - Loading pretrained SD-VAE
    - Encoding images to latent space
    - Decoding latents to pixel space
    - Proper scaling of latents

    Args:
        pretrained_path: HuggingFace model path for the VAE
                        Default: "stabilityai/sd-vae-ft-mse" (SD 1.x VAE fine-tuned with MSE)
        scaling_factor: Latent scaling factor
    """

    def __init__(
        self,
        pretrained_path: str = "stabilityai/sd-vae-ft-mse",
        scaling_factor: float = 0.18215,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor

        try:
            from diffusers import AutoencoderKL
            self.vae = AutoencoderKL.from_pretrained(pretrained_path)
        except ImportError:
            raise ImportError(
                "diffusers is required for VAE. Install with: pip install diffusers"
            )

        # Freeze VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()

    @property
    def latent_channels(self) -> int:
        return self.vae.config.latent_channels  # Typically 4

    @property
    def downsample_factor(self) -> int:
        return 2 ** (len(self.vae.config.block_out_channels) - 1)  # Typically 8

    def get_latent_size(self, image_size: int) -> int:
        """Get the latent spatial size for a given image size."""
        return image_size // self.downsample_factor

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent space.

        Args:
            x: [B, 3, H, W] images in [-1, 1] range

        Returns:
            [B, 4, H/8, W/8] latent codes (scaled)
        """
        posterior = self.vae.encode(x).latent_dist
        z = posterior.sample()
        z = z * self.scaling_factor
        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to pixel space.

        Args:
            z: [B, 4, H/8, W/8] latent codes (scaled)

        Returns:
            [B, 3, H, W] images in [-1, 1] range
        """
        z = z / self.scaling_factor
        x = self.vae.decode(z).sample
        return x

    @torch.no_grad()
    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images using the posterior mean (deterministic).

        Args:
            x: [B, 3, H, W] images in [-1, 1] range

        Returns:
            [B, 4, H/8, W/8] latent codes (scaled)
        """
        posterior = self.vae.encode(x).latent_dist
        z = posterior.mean
        z = z * self.scaling_factor
        return z

    def train(self, mode: bool = True):
        """Keep VAE in eval mode always."""
        super().train(mode)
        self.vae.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode by default."""
        return self.encode(x)
