"""
Drifting DiT (Diffusion Transformer) — modified for single-step generation.

Based on the DiT architecture from "Scalable Diffusion Models with Transformers"
(Peebles & Xie, 2023), but adapted for the Drifting Model paradigm:
- No timestep conditioning (single-step, no diffusion process)
- Class-conditional only via AdaLN
- Direct noise-to-sample mapping

Reference architecture sizes from the paper:
  DiT-L/2: ~463M params (latent space), ~464M params (pixel space)
  - Hidden dim: 1024
  - Depth: 24 blocks
  - Heads: 16
  - Patch size: 2
"""

import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


def modulate(x, shift, scale):
    """Apply adaptive layer normalization modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ─────────────────────────────────────────────────────────────
# Positional Embeddings
# ─────────────────────────────────────────────────────────────

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """Generate 2D sinusoidal positional embeddings."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # (2, grid_size, grid_size)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


# ─────────────────────────────────────────────────────────────
# Label Embedder (class conditioning)
# ─────────────────────────────────────────────────────────────

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


# ─────────────────────────────────────────────────────────────
# Patch Embed
# ─────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """2D image/latent to patch embedding."""
    def __init__(
        self,
        img_size=32,
        patch_size=2,
        in_channels=4,
        embed_dim=1024,
        bias=True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


# ─────────────────────────────────────────────────────────────
# Attention
# ─────────────────────────────────────────────────────────────

class Attention(nn.Module):
    """Multi-head self-attention."""
    def __init__(self, dim, num_heads=16, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv.unbind(0)

        # Use scaled dot-product attention (Flash Attention when available)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


# ─────────────────────────────────────────────────────────────
# MLP (Feed-Forward)
# ─────────────────────────────────────────────────────────────

class Mlp(nn.Module):
    """MLP with GELU activation."""
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# ─────────────────────────────────────────────────────────────
# DiT Block (AdaLN-Zero, class-only conditioning)
# ─────────────────────────────────────────────────────────────

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Modified: conditioned only on class embedding (no timestep).
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
        )
        # AdaLN-Zero modulation: 6 parameters per token
        # (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        """
        Args:
            x: (B, N, D) token features
            c: (B, D) conditioning vector (class embedding)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


# ─────────────────────────────────────────────────────────────
# Final Layer
# ─────────────────────────────────────────────────────────────

class FinalLayer(nn.Module):
    """Final layer of DiT: AdaLN + linear projection to patch."""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ─────────────────────────────────────────────────────────────
# DriftingDiT: Main Generator Model
# ─────────────────────────────────────────────────────────────

class DriftingDiT(nn.Module):
    """
    Drifting Model generator based on DiT architecture.

    Maps noise ε ~ N(0, I) to samples via a single forward pass.
    Conditioned on class labels only (no timestep).

    Args:
        input_size: Spatial size of the input noise (e.g., 32 for 32x32 latent)
        patch_size: Patch size for patchification
        in_channels: Number of input channels (e.g., 4 for SD-VAE latent)
        hidden_size: Transformer hidden dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        num_classes: Number of classes for conditioning
        class_dropout_prob: Dropout probability for CFG training
        learn_sigma: Whether to predict variance (not used in drifting, kept for compatibility)
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=1000,
        class_dropout_prob=0.1,
        learn_sigma=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels  # Same as input
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.num_classes = num_classes

        # Patch embedding
        self.x_embedder = PatchEmbed(
            img_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )
        # Class embedding
        self.y_embedder = LabelEmbedder(
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout_prob=class_dropout_prob,
        )

        num_patches = self.x_embedder.num_patches
        # Fixed positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        # Final layer
        self.final_layer = FinalLayer(
            hidden_size, patch_size, self.out_channels
        )

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights following DiT conventions."""
        # Initialize transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize positional embeddings with sin-cos
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches**0.5),
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        # Initialize patch embed like nn.Linear
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Zero-out adaLN modulation layers (DiT convention)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        Reshape patch tokens back to spatial format.
        x: (B, N, patch_size**2 * C)
        Returns: (B, C, H, W)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(x.shape[0], c, h * p, w * p)
        return imgs

    def forward(self, x, y, force_drop_ids=None):
        """
        Forward pass of DriftingDiT.

        Args:
            x: (B, C, H, W) input noise tensor
            y: (B,) class labels
            force_drop_ids: Optional tensor for forcing CFG dropout

        Returns:
            (B, C, H, W) generated sample
        """
        # Patchify and embed
        x = self.x_embedder(x) + self.pos_embed  # (B, N, D)

        # Class conditioning
        c = self.y_embedder(y, self.training, force_drop_ids)  # (B, D)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # Final layer and unpatchify
        x = self.final_layer(x, c)  # (B, N, p*p*C)
        x = self.unpatchify(x)  # (B, C, H, W)
        return x

    def forward_with_cfg(self, x, y, cfg_scale):
        """
        Forward pass with classifier-free guidance at inference.

        Args:
            x: (B, C, H, W) input noise
            y: (B,) class labels
            cfg_scale: CFG scale factor (1.0 = no guidance)

        Returns:
            (B, C, H, W) guided output
        """
        # Run conditional and unconditional in a single batch
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, y)
        cond_out, uncond_out = model_out.chunk(2, dim=0)
        # CFG formula
        guided = uncond_out + cfg_scale * (cond_out - uncond_out)
        return guided


# ─────────────────────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────────────────────

def DiT_XL_2(**kwargs):
    return DriftingDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DriftingDiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DriftingDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DriftingDiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DriftingDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DriftingDiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DriftingDiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DriftingDiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


DiT_models = {
    "DiT-XL/2": DiT_XL_2,
    "DiT-XL/4": DiT_XL_4,
    "DiT-L/2": DiT_L_2,
    "DiT-L/4": DiT_L_4,
    "DiT-B/2": DiT_B_2,
    "DiT-B/4": DiT_B_4,
    "DiT-S/2": DiT_S_2,
    "DiT-S/4": DiT_S_4,
}
