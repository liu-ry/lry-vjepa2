"""把触觉 token 对齐到冻结 V-JEPA latent 的辅助损失。

主信号是 token 级未来预测（见 ``FutureLatentPredictor``）。相机和触觉没有共享
空间格子，所以这里只对齐帧级摘要。视觉输入会 detach，梯度只进触觉编码器/投影头。
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tactile_encoder import TactileEncoder

__all__ = ["TactileEncoder", "TactileAlignment"]


class TactileAlignment(nn.Module):
    """对齐触觉与 V-JEPA latent 的投影头和损失。"""

    def __init__(self, tactile_dim: int, visual_dim: int, projection_dim: int = 256,
                 temperature: float = 0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.tactile_proj = nn.Sequential(nn.LayerNorm(tactile_dim), nn.Linear(tactile_dim, projection_dim))
        self.visual_proj = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, projection_dim))

    @staticmethod
    def _pool_tokens(z: torch.Tensor) -> torch.Tensor:
        """对空间 token 池化、保留时间：[B,T,N,D] -> [B,T,D]。"""
        if z.ndim == 4:
            return z.mean(dim=2)
        if z.ndim == 3:
            return z
        return z[:, None, :]

    @staticmethod
    def _pool(z: torch.Tensor) -> torch.Tensor:
        # 接受 [B,D] / [B,T,D] / [B,T,N,D]，对 token 和时间都做池化。
        z = TactileAlignment._pool_tokens(z)
        while z.ndim > 2:
            z = z.mean(dim=-2)
        return z

    def info_nce(self, tactile: torch.Tensor, visual: torch.Tensor,
                 gather_distributed: bool = False) -> torch.Tensor:
        t = F.normalize(self.tactile_proj(self._pool(tactile)), dim=-1)
        v = F.normalize(self.visual_proj(self._pool(visual.detach())), dim=-1)
        if t.shape[0] < 2:
            # 单样本 batch 没有负例，InfoNCE 退化为 0。
            return t.sum() * 0.0
        if gather_distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
            world = torch.distributed.get_world_size()
            gathered_t = [torch.zeros_like(t) for _ in range(world)]
            gathered_v = [torch.zeros_like(v) for _ in range(world)]
            torch.distributed.all_gather(gathered_t, t.detach())
            torch.distributed.all_gather(gathered_v, v.detach())
            # 负例用全局 gather，本地触觉梯度保留。
            gathered_t[torch.distributed.get_rank()] = t
            gathered_v[torch.distributed.get_rank()] = v
            t_all, v_all = torch.cat(gathered_t), torch.cat(gathered_v)
            logits = t @ v_all.transpose(0, 1) / self.temperature
            labels = torch.arange(t.shape[0], device=t.device) + torch.distributed.get_rank() * t.shape[0]
            logits_t = t_all @ v.transpose(0, 1) / self.temperature
            labels_t = torch.arange(t.shape[0], device=t.device) + torch.distributed.get_rank() * t.shape[0]
            return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits_t.transpose(0, 1), labels_t))
        logits = t @ v.transpose(0, 1) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))

    @staticmethod
    def latent_regression(tactile: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        t = F.normalize(TactileAlignment._pool(tactile), dim=-1)
        v = F.normalize(TactileAlignment._pool(visual.detach()), dim=-1)
        return F.mse_loss(t, v)

    @staticmethod
    def temporal_consistency(tactile: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        # 需要时间维。单帧 batch 返回 0，避免无效差分。
        if tactile.ndim < 3 or visual.ndim < 3 or tactile.shape[1] < 2 or visual.shape[1] < 2:
            return tactile.new_zeros(())
        t = TactileAlignment._pool_tokens(tactile)
        v = TactileAlignment._pool_tokens(visual.detach())
        return F.mse_loss(t[:, 1:] - t[:, :-1], v[:, 1:] - v[:, :-1])

    def forward(self, tactile: torch.Tensor, visual: torch.Tensor,
                lambda_global: float = 1.0, lambda_latent: float = 1.0,
                lambda_temporal: float = 0.1, gather_distributed: bool = False) -> Tuple[torch.Tensor, dict]:
        l_global = self.info_nce(tactile, visual, gather_distributed=gather_distributed)
        # 两模态宽度不同时，帧级回归走共享投影。
        if tactile.shape[-1] == visual.shape[-1]:
            l_latent = self.latent_regression(tactile, visual)
        else:
            t = F.normalize(self.tactile_proj(self._pool(tactile)), dim=-1)
            v = F.normalize(self.visual_proj(self._pool(visual.detach())), dim=-1)
            l_latent = F.mse_loss(t, v)
        if tactile.shape[-1] == visual.shape[-1]:
            l_temp = self.temporal_consistency(tactile, visual)
        else:
            t_seq = F.normalize(self.tactile_proj(self._pool_tokens(tactile)), dim=-1)
            v_seq = F.normalize(self.visual_proj(self._pool_tokens(visual.detach())), dim=-1)
            if t_seq.shape[1] < 2 or v_seq.shape[1] < 2:
                l_temp = tactile.new_zeros(())
            else:
                l_temp = F.mse_loss(t_seq[:, 1:] - t_seq[:, :-1],
                                    v_seq[:, 1:] - v_seq[:, :-1])
        total = lambda_global * l_global + lambda_latent * l_latent + lambda_temporal * l_temp
        return total, {"loss_global": l_global.detach(), "loss_latent": l_latent.detach(), "loss_temporal": l_temp.detach()}
