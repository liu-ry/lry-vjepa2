"""输出 V-JEPA 风格 token 的时空触觉编码器。

结构是 tubelet ViT：3D patch 嵌入 + V-JEPA 2.1 同一套 Transformer Block（可选 3D RoPE）。
输出是 ``[B, T, N, D]`` token，不做全局池化，以便多模态预测器按冻结视觉编码器
同样的时空布局消费。

向量 / taxel 观测用每个 tubelet 上的 MLP，再走同一套 Transformer。
``taxel_grid`` 会把展平的 taxel 图 reshape 成单通道图像，再走图像通路。
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.vjepa_2_1.models.utils.modules import Block
from app.vjepa_2_1.models.utils.patch_embed import PatchEmbed3D


def _as_hw(size: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(size, int):
        return size, size
    if len(size) != 2:
        raise ValueError(f"img_size must be int or (H, W), got {size}")
    return int(size[0]), int(size[1])


def _default_heads(embed_dim: int, num_heads: Optional[int]) -> int:
    if num_heads is not None:
        if embed_dim % num_heads:
            raise ValueError(f"embed_dim {embed_dim} is not divisible by num_heads {num_heads}")
        return num_heads
    for heads in (8, 6, 4, 2):
        if embed_dim % heads == 0 and embed_dim // heads >= 8:
            return heads
    raise ValueError(f"embed_dim {embed_dim} needs an explicit num_heads")


class TactileEncoder(nn.Module):
    """Tubelet ViT 触觉编码器。

    Args:
        embed_dim: token 宽度。可以和 V-JEPA 不同，预测器会分模态投影。
        tubelet_size: 时间 patch 大小。完整触觉 clip 可以和冻结 V-JEPA 对齐；
            当前训练入口的在线单帧触觉路径显式使用 ``1``。
        img_size: 若设置，会把图像观测在空间上 resize 到该尺寸。
        taxel_grid: 可选 ``(H, W)``，把长度为 H*W 的展平 taxel 图当图像用。
    """

    def __init__(
        self,
        embed_dim: int = 384,
        in_channels: int = 3,
        patch_size: int = 16,
        tubelet_size: int = 1,
        img_size: Optional[Union[int, Sequence[int]]] = None,
        num_frames: int = 16,
        depth: int = 6,
        num_heads: Optional[int] = None,
        mlp_ratio: float = 4.0,
        input_dim: Optional[int] = None,
        taxel_grid: Optional[Sequence[int]] = None,
        use_rope: bool = True,
        drop_path: float = 0.0,
        backbone: Optional[str] = None,
        pretrained: bool = False,
    ):
        super().__init__()
        if backbone is not None and str(backbone).lower().startswith("resnet"):
            raise ValueError(
                "ResNet tactile backbones were removed. The encoder is a spatiotemporal "
                "ViT; drop tactile.backbone from the config."
            )
        del pretrained
        if tubelet_size <= 0:
            raise ValueError("tubelet_size must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.in_channels = in_channels
        self.input_dim = input_dim
        self.use_rope = use_rope
        self.img_size = _as_hw(img_size) if img_size is not None else None
        self.taxel_grid = tuple(taxel_grid) if taxel_grid is not None else None
        heads = _default_heads(embed_dim, num_heads)
        self.num_heads = heads

        if self.taxel_grid is not None:
            if len(self.taxel_grid) != 2 or min(self.taxel_grid) <= 0:
                raise ValueError("taxel_grid must be a positive (H, W)")
            self.in_channels = 1
            input_dim = None
            self.input_dim = None

        grid_h = (self.img_size[0] // patch_size) if self.img_size is not None else 14
        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    use_rope=use_rope,
                    grid_size=max(grid_h, 1),
                    patch_size=patch_size,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        if input_dim is None:
            self.patch_embed = PatchEmbed3D(
                patch_size=patch_size,
                tubelet_size=tubelet_size,
                in_chans=self.in_channels,
                embed_dim=embed_dim,
            )
            self.vector_proj = None
        else:
            self.patch_embed = None
            self.vector_proj = nn.Sequential(
                nn.Linear(input_dim * tubelet_size, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv3d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def _resize_images(self, x: torch.Tensor) -> torch.Tensor:
        if self.img_size is None:
            return x
        height, width = x.shape[-2:]
        if (height, width) == self.img_size:
            return x
        batch, channels, time, _, _ = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
        x = F.interpolate(x, size=self.img_size, mode="bilinear", align_corners=False)
        height, width = self.img_size
        return x.view(batch, time, channels, height, width).permute(0, 2, 1, 3, 4)

    def _run_blocks(self, tokens: torch.Tensor, time: int, height: int, width: int) -> torch.Tensor:
        for block in self.blocks:
            tokens, _ = block(tokens, T=time, H_patches=height, W_patches=width)
        return self.norm(tokens)

    def _encode_image(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_embed is None:
            raise ValueError("vector tactile encoder received image input")
        if x.ndim == 4:
            x = x.unsqueeze(2)
        elif x.ndim == 5:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        else:
            raise ValueError(f"expected [B,C,H,W] or [B,T,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} tactile channels, got {x.shape[1]}")
        x = self._resize_images(x)
        _, _, time, height, width = x.shape
        if time % self.tubelet_size:
            raise ValueError(
                f"tactile length {time} must be divisible by tubelet_size {self.tubelet_size}"
            )
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"tactile spatial size {(height, width)} must be divisible by patch_size {self.patch_size}"
            )
        tokens = self.patch_embed(x)
        time_out = time // self.tubelet_size
        height_p, width_p = height // self.patch_size, width // self.patch_size
        tokens = self._run_blocks(tokens, time_out, height_p, width_p)
        return tokens.view(x.shape[0], time_out, height_p * width_p, self.embed_dim)

    def _encode_vector(self, x: torch.Tensor) -> torch.Tensor:
        if self.vector_proj is None:
            raise ValueError("image tactile encoder received vector input")
        if x.ndim != 3:
            raise ValueError(f"expected [B,T,F] vector tactile, got {tuple(x.shape)}")
        batch, time, feat = x.shape
        if feat != self.input_dim:
            raise ValueError(f"expected vector dim {self.input_dim}, got {feat}")
        if time % self.tubelet_size:
            raise ValueError(
                f"tactile length {time} must be divisible by tubelet_size {self.tubelet_size}"
            )
        time_out = time // self.tubelet_size
        x = x.view(batch, time_out, self.tubelet_size * feat)
        tokens = self.vector_proj(x)
        tokens = self._run_blocks(tokens, time_out, 1, 1)
        return tokens.view(batch, time_out, 1, self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3 and self.taxel_grid is not None:
            batch, time, feat = x.shape
            grid_h, grid_w = self.taxel_grid
            if feat != grid_h * grid_w:
                raise ValueError(
                    f"taxel vector dim {feat} does not match taxel_grid {self.taxel_grid}"
                )
            x = x.view(batch, time, 1, grid_h, grid_w)
        if x.ndim in (4, 5):
            return self._encode_image(x)
        if x.ndim == 3:
            return self._encode_vector(x)
        raise ValueError(f"expected [B,T,F], [B,T,C,H,W], or [B,C,H,W], got {tuple(x.shape)}")
