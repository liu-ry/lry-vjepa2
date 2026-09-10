"""对未来 V-JEPA / 触觉 latent 做 token 级预测。

输入是冻结 V-JEPA 2.1 的视觉 context token 和 ``TactileEncoder`` 的触觉 token，
后面拼上可学习 mask token，预测各模态完整时空 latent。序列末尾另有一组
condition query，attend context 与未来 token，再投影成短序列给下游
flow matching。视觉/状态可以有多帧历史；触觉 context 可以是边界单帧以便低延迟。
空间 token 从不池化：输出格子与各模态 context 格子一致，预测器才能待在 V-JEPA 隐空间里。
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.vjepa_2_1.models.utils.modules import Block


def _fit_pos(table: torch.Tensor, count: int) -> torch.Tensor:
    """把 ``[1, M, D]`` 位置表裁切或线性插值到 ``count``。"""
    if count <= table.shape[1]:
        return table[:, :count]
    return F.interpolate(table.transpose(1, 2), size=count, mode="linear", align_corners=False).transpose(1, 2)


class FutureLatentPredictor(nn.Module):
    """JEPA 风格的多模态未来 latent 预测器。

    Transformer 内顺序为
    ``[触觉 context | 视觉 context | 可选 state | 未来触觉 query | 未来视觉 query | condition query]``。
    注意力是 prefix encoder-decoder：context 只看 context；未来 query 彼此自注意力
    并 cross-attend context；condition query 看整条序列。context 看不见 query 槽位，
    历史编码不会混进「我在问 t+k」。condition query 没有时间/空间位置编码，是读出用的，不是格子。
    """

    def __init__(
        self,
        dim: int = 1024,
        hidden_dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        horizon: int = 4,
        visual_dim: Optional[int] = None,
        tactile_dim: Optional[int] = None,
        visual_out_dim: Optional[int] = None,
        tactile_out_dim: Optional[int] = None,
        max_context: int = 32,
        max_visual_tokens: int = 256,
        max_tactile_tokens: int = 256,
        drop_path: float = 0.0,
        future_offsets: Optional[Sequence[int]] = None,
        max_future_offset: Optional[int] = None,
        state_dim: Optional[int] = None,
        num_condition_tokens: int = 16,
        condition_out_dim: Optional[int] = None,
    ):
        super().__init__()
        if heads <= 0 or hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")
        if max_context <= 0 or max_visual_tokens <= 0 or max_tactile_tokens <= 0:
            raise ValueError("max_context and max token counts must be positive")

        self.dim = dim
        # ``horizon`` 是连续 rollout 的容量。稀疏 waypoint 计划可以更少
        # （例如 5, 10, ..., 30 共六个 query），配置的 offset 个数不必等于该容量。
        self.horizon = horizon
        self.visual_dim = visual_dim or dim
        self.tactile_dim = tactile_dim or dim
        self.visual_out_dim = visual_out_dim or self.visual_dim
        self.tactile_out_dim = tactile_out_dim or self.tactile_dim
        self.max_context = max_context
        self.hidden_dim = hidden_dim
        configured_offsets = range(1, horizon + 1) if future_offsets is None else future_offsets
        self.future_offsets = tuple(int(x) for x in configured_offsets)
        if not self.future_offsets or any(x <= 0 for x in self.future_offsets):
            raise ValueError("future_offsets must contain at least one positive offset")
        if tuple(sorted(set(self.future_offsets))) != self.future_offsets:
            raise ValueError("future_offsets must be sorted and unique")
        configured_max_offset = max(self.future_offsets)
        self.max_future_offset = configured_max_offset if max_future_offset is None else int(max_future_offset)
        if self.max_future_offset < configured_max_offset:
            raise ValueError("max_future_offset cannot be smaller than a configured future offset")
        if self.max_future_offset <= 0:
            raise ValueError("max_future_offset must be positive")
        if num_condition_tokens < 0:
            raise ValueError("num_condition_tokens must be non-negative")

        self.visual_in = nn.Linear(self.visual_dim, hidden_dim)
        self.tactile_in = nn.Linear(self.tactile_dim, hidden_dim)
        self.state_in = nn.Linear(state_dim, hidden_dim) if state_dim is not None else None
        self.visual_query = nn.Parameter(torch.randn(1, 1, 1, hidden_dim) * 0.02)
        self.tactile_query = nn.Parameter(torch.randn(1, 1, 1, hidden_dim) * 0.02)
        self.time_pos = nn.Parameter(torch.randn(1, max_context + self.max_future_offset, hidden_dim) * 0.02)
        self.visual_pos = nn.Parameter(torch.randn(1, max_visual_tokens, hidden_dim) * 0.02)
        self.tactile_pos = nn.Parameter(torch.randn(1, max_tactile_tokens, hidden_dim) * 0.02)
        self.modality = nn.Parameter(torch.randn(1, 1, 1, 2, hidden_dim) * 0.02)
        self.num_condition_tokens = int(num_condition_tokens)
        self.condition_out_dim = int(condition_out_dim) if condition_out_dim is not None else hidden_dim
        if self.num_condition_tokens > 0:
            self.condition_query = nn.Parameter(
                torch.randn(1, self.num_condition_tokens, hidden_dim) * 0.02
            )
            self.condition_modality = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            self.out_condition = nn.Linear(hidden_dim, self.condition_out_dim)
        else:
            self.condition_query = None
            self.condition_modality = None
            self.out_condition = None

        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(dim=hidden_dim, num_heads=heads, mlp_ratio=4.0, qkv_bias=True, drop_path=dpr[i])
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_visual = nn.Linear(hidden_dim, self.visual_out_dim)
        self.out_tactile = nn.Linear(hidden_dim, self.tactile_out_dim)
        self.apply(self._init_weights)
        self._rescale_blocks()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def _rescale_blocks(self) -> None:
        for layer_id, block in enumerate(self.blocks):
            block.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    @staticmethod
    def _as_tokens(z: torch.Tensor, name: str) -> torch.Tensor:
        if z.ndim == 3:
            return z.unsqueeze(2)
        if z.ndim == 4:
            return z
        raise ValueError(f"{name} must be [B,T,N,D] or [B,T,D], got {tuple(z.shape)}")

    def _stamp(self, tokens: torch.Tensor, t0: int, spatial: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
        _, time, count, _ = tokens.shape
        if time <= 0 or count <= 0:
            raise ValueError("token sequences must contain at least one time step and token")
        if t0 + time > self.time_pos.shape[1]:
            raise ValueError(
                f"time index {t0 + time} exceeds positional table {self.time_pos.shape[1]}; "
                "increase max_context"
            )
        return (
            tokens
            + self.time_pos[:, t0 : t0 + time, None, :]
            + _fit_pos(spatial, count)[:, None, :, :]
            + modality
        )

    def _stamp_indices(self, tokens: torch.Tensor, indices: Sequence[int],
                       spatial: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
        if len(indices) != tokens.shape[1]:
            raise ValueError("time index count must match token sequence length")
        if tokens.shape[2] <= 0:
            raise ValueError("token sequences must contain at least one spatial token")
        idx = torch.as_tensor(indices, device=tokens.device, dtype=torch.long)
        if idx.min() < 0 or idx.max() >= self.time_pos.shape[1]:
            raise ValueError("future time index exceeds positional table")
        return tokens + self.time_pos[:, idx, None, :] + _fit_pos(spatial, tokens.shape[2])[:, None] + modality

    def _resolve_future_offsets(self, future_offsets: Optional[Sequence[int]]) -> Tuple[int, ...]:
        offsets = self.future_offsets if future_offsets is None else tuple(int(x) for x in future_offsets)
        if not offsets or any(x <= 0 for x in offsets):
            raise ValueError("future_offsets must contain positive integers")
        if tuple(sorted(offsets)) != offsets or len(set(offsets)) != len(offsets):
            raise ValueError("future_offsets must be unique and sorted")
        if offsets[-1] > self.max_future_offset:
            raise ValueError(
                f"future offset {offsets[-1]} exceeds predictor capacity {self.max_future_offset}"
            )
        return offsets

    def _condition_tokens(self, batch: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """可学习读出 query：身份向量 + condition 模态编码，不加时间/空间位置。"""
        return (self.condition_query.to(device=device, dtype=dtype)
                + self.condition_modality.to(device=device, dtype=dtype)).expand(batch, -1, -1)

    def _attention_mask(self, context_len: int, future_len: int, cond_len: int,
                        dtype: torch.dtype, device: torch.device) -> Optional[torch.Tensor]:
        """``[context | future | condition]`` 上的 prefix encoder-decoder mask。

        * context（encoder）只看 context。
        * 未来 query（decoder）看 context 和其他未来 query。
        * condition query 看所有 token，包括彼此。
        * context 和未来 query 不能看 condition query。

        加性 ``-inf`` 让 SDPA 与手工注意力一致。没有 query、无需屏蔽时返回 ``None``。
        """
        if context_len < 0 or future_len < 0 or cond_len < 0:
            raise ValueError("token segment lengths must be non-negative")
        total = context_len + future_len + cond_len
        if total == 0:
            return None
        if future_len == 0 and cond_len == 0:
            return None
        mask = torch.zeros(total, total, device=device, dtype=dtype)
        blocked = torch.finfo(dtype).min
        mask[:context_len, context_len:] = blocked
        if cond_len:
            mask[context_len:context_len + future_len, context_len + future_len:] = blocked
        return mask

    def forward(self, visual: torch.Tensor, tactile: torch.Tensor,
                state: Optional[torch.Tensor] = None,
                future_offsets: Optional[Sequence[int]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """为每个配置的未来 offset 预测一个 latent waypoint。

        Args:
            visual: context token ``[B, T, Nv, Dv]``（或 ``[B, T, Dv]``）。
            tactile: context token ``[B, T, Nt, Dh]``（或 ``[B, T, Dh]``）。
            视觉有多帧时也允许触觉 ``T=1``，视为 context 最后一拍的触觉观测。
            state: 可选本体状态 ``[B, T, Ds]``。
            future_offsets: 可选运行时 query 偏移，个数和间隔可变，不超过配置的最大 offset。

        Returns:
            ``(pred_visual, pred_tactile, condition)``，形状分别为
            ``[B, len(future_offsets), Nv, Dv]``、
            ``[B, len(future_offsets), Nt, Dh]``、
            ``[B, num_condition_tokens, condition_out_dim]``
            （``num_condition_tokens=0`` 时 condition 为 ``None``）。
        """
        visual = self._as_tokens(visual, "visual")
        tactile = self._as_tokens(tactile, "tactile")
        batch, context, n_visual, _ = visual.shape
        if tactile.shape[0] != batch:
            raise ValueError("visual and tactile batch dimensions must match")
        if visual.shape[-1] != self.visual_dim:
            raise ValueError(
                f"visual latent width {visual.shape[-1]} does not match configured {self.visual_dim}"
            )
        if tactile.shape[-1] != self.tactile_dim:
            raise ValueError(
                f"tactile latent width {tactile.shape[-1]} does not match configured {self.tactile_dim}"
            )
        tactile_time = tactile.shape[1]
        if tactile_time not in (1, context):
            raise ValueError(
                "tactile context must contain either one current frame or the "
                f"same number of frames as visual context; got {tactile_time} vs {context}"
            )
        n_tactile = tactile.shape[2]
        offsets = self._resolve_future_offsets(future_offsets)
        num_future = len(offsets)
        if context == 0:
            raise ValueError("predictor requires a non-empty context")
        if context > self.max_context:
            raise ValueError(f"context {context} exceeds max_context {self.max_context}")

        if self.state_in is not None:
            if state is None or state.ndim != 3 or state.shape[:2] != (batch, context):
                raise ValueError("state must be [B,T,state_dim] when state_dim is configured")
            state_ctx = self.state_in(state).unsqueeze(2)
        elif state is not None:
            raise ValueError("state was provided but predictor was built without state_dim")
        visual_ctx = self._stamp(
            self.visual_in(visual), 0, self.visual_pos, self.modality[..., 0, :]
        )
        tactile_ctx = self.tactile_in(tactile)
        tactile_t0 = 0
        if tactile_time == 1 and context > 1:
            # 单帧触觉对齐到视觉/状态历史的终点，不复制到更早的时间步。
            tactile_t0 = context - 1
        tactile_ctx = self._stamp(
            tactile_ctx, tactile_t0, self.tactile_pos, self.modality[..., 1, :]
        )
        # offset 从 context 边界起算：offset=1 是最后一帧观测之后的第一帧。
        # 因此可以用 [5, 10, ..., 30] 这种稀疏 waypoint，而不把它们当成连续帧。
        future_indices = [context + offset - 1 for offset in offsets]
        visual_q = self._stamp_indices(
            self.visual_query.expand(batch, num_future, n_visual, -1),
            future_indices,
            self.visual_pos,
            self.modality[..., 0, :],
        )
        tactile_q = self._stamp_indices(
            self.tactile_query.expand(batch, num_future, n_tactile, -1),
            future_indices,
            self.tactile_pos,
            self.modality[..., 1, :],
        )

        per_frame = n_tactile + n_visual
        # 视觉和状态占满全部 context 时刻。触觉可能只有终点一帧，因此各流单独 flatten，
        # 不要沿共享时间维拼接。
        context_parts = [tactile_ctx.flatten(1, 2), visual_ctx.flatten(1, 2)]
        if self.state_in is not None:
            state_ctx = self._stamp(state_ctx, 0, self.tactile_pos, self.modality[..., 1, :])
            context_parts.append(state_ctx.flatten(1, 2))
        context_tokens = torch.cat(context_parts, dim=1)
        future_tokens = torch.cat([tactile_q, visual_q], dim=2).flatten(1, 2)
        parts = [context_tokens, future_tokens]
        if self.num_condition_tokens > 0:
            parts.append(self._condition_tokens(batch, visual_ctx.dtype, visual_ctx.device))
        tokens = torch.cat(parts, dim=1)
        attn_mask = self._attention_mask(
            context_tokens.shape[1],
            future_tokens.shape[1],
            self.num_condition_tokens,
            visual_ctx.dtype,
            visual_ctx.device,
        )
        for block in self.blocks:
            tokens, _ = block(tokens, attn_mask=attn_mask)
        encoded = self.norm(tokens)
        future_start = context_tokens.shape[1]
        future_end = future_start + future_tokens.shape[1]
        future = encoded[:, future_start:future_end].view(batch, num_future, per_frame, self.hidden_dim)
        pred_tactile = self.out_tactile(future[:, :, :n_tactile])
        pred_visual = self.out_visual(future[:, :, n_tactile:])
        if self.num_condition_tokens <= 0:
            return pred_visual, pred_tactile, None
        condition = self.out_condition(encoded[:, future_end:])
        return pred_visual, pred_tactile, condition

    def rollout(self, visual: torch.Tensor, tactile: torch.Tensor, steps: int,
                window: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """闭环外推 ``steps`` 帧连续 latent。

        一次 ``forward`` 只吐出 ``self.horizon`` 帧连续结果。更长时域把最近 ``window``
        帧预测滑回 context。这是推理展开，不是训练方式。
        """
        visual = self._as_tokens(visual, "visual")
        tactile = self._as_tokens(tactile, "tactile")
        if steps <= 0:
            raise ValueError("steps must be positive")
        if self.future_offsets != tuple(range(1, self.horizon + 1)):
            raise ValueError("rollout requires contiguous future_offsets; sparse waypoints are not autoregressive frames")
        if visual.shape[:2] != tactile.shape[:2]:
            raise ValueError(
                f"visual and tactile context shapes differ: {tuple(visual.shape)} vs {tuple(tactile.shape)}"
            )
        window = visual.shape[1] if window is None else window
        if window <= 0:
            raise ValueError("window must be positive")
        if window > self.max_context:
            raise ValueError(f"window {window} exceeds max_context {self.max_context}")
        ctx_v, ctx_h = visual, tactile
        chunks_v, chunks_h = [], []
        remain = steps
        while remain > 0:
            ctx_v = ctx_v[:, -window:]
            ctx_h = ctx_h[:, -window:]
            pred_v, pred_h, _ = self.forward(ctx_v, ctx_h)
            take = min(self.horizon, remain)
            chunks_v.append(pred_v[:, :take])
            chunks_h.append(pred_h[:, :take])
            ctx_v = torch.cat([ctx_v, pred_v[:, :take]], dim=1)
            ctx_h = torch.cat([ctx_h, pred_h[:, :take]], dim=1)
            remain -= take
        return torch.cat(chunks_v, dim=1), torch.cat(chunks_h, dim=1)
