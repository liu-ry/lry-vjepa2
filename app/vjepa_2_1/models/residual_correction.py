"""在线根据触觉隐空间修正动作 chunk。

世界模型会预测未来 waypoint 上的触觉 latent。新触觉图到达时，调用方选出对应
waypoint，只把尚未执行的动作后缀送进来。已经发给机器人的动作不会被改。
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentResidualController(nn.Module):
    """把预测/观测 latent 的差异映射成动作残差。

    MLP 只看 ``(predicted, observed, error)``。剩余动作只是残差加到的后缀，
    不是网络输入。``forward`` 接受单个 waypoint ``[B,D]``，或长度与动作后缀
    相同的序列。空间格子 ``[B,T,N,D]`` 会做 mean pool；``correct_at`` 按已经
    经过的 raw-frame offset 从预测的 ``[B,K,N,D]`` 里选出对应 waypoint。
    """

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 256,
                 max_residual: Optional[float] = None):
        super().__init__()
        if latent_dim <= 0 or action_dim <= 0 or hidden_dim <= 0:
            raise ValueError("latent_dim, action_dim, and hidden_dim must be positive")
        if max_residual is not None and max_residual <= 0:
            raise ValueError("max_residual must be positive")
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.max_residual = max_residual
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.action_dim),
        )

    @staticmethod
    def _as_sequence(latent: torch.Tensor, name: str) -> torch.Tensor:
        """把 ``[B,D]`` / ``[B,T,D]`` / ``[B,T,N,D]`` 转成 ``[B,T,D]``。"""
        if latent.ndim == 2:
            return latent.unsqueeze(1)
        if latent.ndim == 3:
            return latent
        if latent.ndim == 4:
            return latent.mean(dim=2)
        raise ValueError(f"{name} must be [B,D], [B,T,D], or [B,T,N,D], got {tuple(latent.shape)}")

    def _residual(self, predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        if predicted.shape != observed.shape:
            raise ValueError("predicted and observed latent shapes must match")
        if predicted.shape[-1] != self.latent_dim:
            raise ValueError(
                f"latent width {predicted.shape[-1]} does not match configured {self.latent_dim}"
            )
        error = observed - predicted
        residual = self.net(torch.cat((predicted, observed, error), dim=-1))
        if self.max_residual is not None:
            residual = self.max_residual * torch.tanh(residual / self.max_residual)
        return residual

    def forward(self, predicted: torch.Tensor, observed: torch.Tensor,
                remaining_actions: torch.Tensor) -> torch.Tensor:
        """返回修正后的剩余动作后缀。

        MLP 不吃 ``remaining_actions``，它们只是残差加到的后缀。单个 waypoint
        的 latent 会广播到每一个剩余动作；若传入序列，长度必须与后缀一致。
        """
        if remaining_actions.ndim != 3:
            raise ValueError(f"remaining_actions must be [B,T,A], got {tuple(remaining_actions.shape)}")
        if remaining_actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"action width {remaining_actions.shape[-1]} does not match configured {self.action_dim}"
            )
        predicted = self._as_sequence(predicted, "predicted")
        observed = self._as_sequence(observed, "observed")
        if predicted.shape[0] != remaining_actions.shape[0] or observed.shape[0] != remaining_actions.shape[0]:
            raise ValueError("latent and action batch dimensions must match")
        if observed.shape[1] not in (1, predicted.shape[1]):
            raise ValueError("observed waypoint count must be one or match predicted waypoint count")
        if observed.shape[1] == 1 and predicted.shape[1] != 1:
            observed = observed.expand(-1, predicted.shape[1], -1)
        residual = self._residual(predicted, observed)
        if residual.shape[1] == 1:
            residual = residual.expand(-1, remaining_actions.shape[1], -1)
        elif residual.shape[1] != remaining_actions.shape[1]:
            raise ValueError("latent waypoint count must be one or match remaining action count")
        return remaining_actions + residual

    def correct_at(self, predicted_future: torch.Tensor, observed_current: torch.Tensor,
                   remaining_actions: torch.Tensor, elapsed_offset: int,
                   future_offsets: Sequence[int]) -> torch.Tensor:
        """用 ``elapsed_offset`` 处的预测去修正后缀。

        ``future_offsets`` 是相对 context 边界的 raw-frame 偏移，必须与
        ``predicted_future`` 的第二维一一对应。必须精确匹配，避免拿错时刻的预测。
        """
        offsets = tuple(int(x) for x in future_offsets)
        if predicted_future.ndim == 2:
            expected = 1
        elif predicted_future.ndim in (3, 4):
            expected = predicted_future.shape[1]
        else:
            raise ValueError("predicted_future must be [B,D], [B,K,D], or [B,K,N,D]")
        if len(offsets) != expected:
            raise ValueError("future_offsets must match the predicted waypoint count")
        if tuple(sorted(set(offsets))) != offsets or any(x <= 0 for x in offsets):
            raise ValueError("future_offsets must be sorted, unique, and positive")
        try:
            index = offsets.index(int(elapsed_offset))
        except ValueError as exc:
            raise ValueError(
                f"elapsed_offset={elapsed_offset} is not represented in future_offsets={offsets}"
            ) from exc
        if predicted_future.ndim == 4:
            # 从 [B,K,N,D] 取出一帧得到 [B,N,D]，这是一张空间格子，不是 N 个时间步。
            # 保持这个含义，直到 ``forward`` 对空间维做池化。
            selected = predicted_future[:, index].unsqueeze(1)
            if observed_current.ndim == 3:
                observed_current = observed_current.unsqueeze(1)
        else:
            selected = predicted_future if predicted_future.ndim == 2 else predicted_future[:, index]
        return self(selected, observed_current, remaining_actions)

    def training_loss(self, predicted: torch.Tensor, observed: torch.Tensor,
                      remaining_actions: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
        """用修正后的动作目标做示范监督损失。"""
        if target_actions.shape != remaining_actions.shape:
            raise ValueError("target_actions and remaining_actions must have the same shape")
        corrected = self(predicted, observed, remaining_actions)
        return F.mse_loss(corrected, target_actions)


__all__ = ["LatentResidualController"]
