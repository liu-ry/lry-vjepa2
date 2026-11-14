"""Self-supervised helpers for independently pretraining the tactile encoder."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TactilePatchDecoder(nn.Module):
    """Decode tactile token grids back to the encoder input image.

    Stage A reconstructs ``[current, current - previous]``.  This gives the
    tactile encoder a geometry-preserving objective before its latent space is
    frozen for future-dynamics learning.
    """

    def __init__(
        self,
        embed_dim: int,
        out_channels: int,
        patch_size: int,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        if embed_dim <= 0 or out_channels <= 0 or patch_size <= 0:
            raise ValueError("decoder dimensions must be positive")
        hidden_dim = int(hidden_dim or embed_dim)
        self.out_channels = int(out_channels)
        self.patch_size = int(patch_size)
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels * patch_size * patch_size),
        )

    def forward(self, tokens: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError(f"tokens must be [B,T,N,D], got {tuple(tokens.shape)}")
        batch, time, count, _ = tokens.shape
        height, width = int(image_size[0]), int(image_size[1])
        p = self.patch_size
        if height % p or width % p:
            raise ValueError("image size must be divisible by patch size")
        if count != (height // p) * (width // p):
            raise ValueError(
                f"token count {count} does not match patch grid "
                f"{(height // p, width // p)}"
            )
        patches = self.net(tokens.reshape(batch * time, count, -1))
        patches = patches.view(
            batch * time, count, self.out_channels, p, p
        )
        patches = patches.permute(0, 2, 3, 4, 1).reshape(
            batch * time, self.out_channels * p * p, count
        )
        image = F.fold(
            patches,
            output_size=(height, width),
            kernel_size=p,
            stride=p,
        )
        return image.view(batch, time, self.out_channels, height, width)


__all__ = ["TactilePatchDecoder"]
