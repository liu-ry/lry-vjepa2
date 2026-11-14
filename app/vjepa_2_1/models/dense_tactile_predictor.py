"""Causal dense-horizon tactile latent predictor."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class DenseTactilePredictor(nn.Module):
    """Predict every future tactile latent in a contiguous horizon."""

    def __init__(
        self,
        tactile_dim: int,
        visual_dim: int,
        state_dim: int,
        hidden_dim: int = 384,
        depth: int = 2,
        heads: int = 6,
        max_time: int = 40,
        num_hands: int = 2,
        future_offsets: Sequence[int] = tuple(range(1, 31)),
        visual_future_offsets: Sequence[int] = tuple(range(2, 31, 2)),
        num_condition_tokens: int = 4,
        drop_path: float = 0.05,
    ):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.tactile_dim = int(tactile_dim)
        self.visual_dim = int(visual_dim)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_hands = int(num_hands)
        self.future_offsets = tuple(int(x) for x in future_offsets)
        self.visual_future_offsets = tuple(int(x) for x in visual_future_offsets)
        self.num_condition_tokens = int(num_condition_tokens)
        if not self.future_offsets:
            raise ValueError("future_offsets cannot be empty")
        if not self.visual_future_offsets:
            raise ValueError("visual_future_offsets cannot be empty")
        if self.num_condition_tokens <= 0:
            raise ValueError("num_condition_tokens must be positive")
        if max(self.future_offsets) >= int(max_time):
            raise ValueError("future offset exceeds time embedding table")

        h = self.hidden_dim
        self.tactile_in = nn.Linear(2 * self.tactile_dim, h)
        self.visual_in = nn.Linear(self.visual_dim, h)
        self.state_in = nn.Linear(self.state_dim, h)
        self.hand_embedding = nn.Parameter(torch.randn(1, num_hands, h) * 0.02)
        self.time_embedding = nn.Parameter(torch.randn(1, int(max_time), h) * 0.02)
        self.visual_embedding = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.state_embedding = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        # A zero residual gate also zeroes its incoming gradient.  Start small
        # enough to preserve the touch-only curriculum, but non-zero so vision
        # and state can actually learn to contribute after modality warmup.
        self.visual_gate = nn.Parameter(torch.full((1,), 0.05))
        self.state_gate = nn.Parameter(torch.full((1,), 0.05))
        self.detach_auxiliary_paths = False
        self.use_visual = True
        self.use_state = True
        layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=heads,
            dim_feedforward=4 * h,
            dropout=drop_path,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.context_norm = nn.LayerNorm(h)
        self.visual_fusion = nn.MultiheadAttention(
            h, heads, dropout=drop_path, batch_first=True
        )
        self.visual_fusion_norm = nn.LayerNorm(h)
        self.state_fusion = nn.MultiheadAttention(
            h, heads, dropout=drop_path, batch_first=True
        )
        self.state_fusion_norm = nn.LayerNorm(h)
        self.future_query = nn.Parameter(torch.randn(1, h) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            h, heads, dropout=drop_path, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(h)
        self.future_head = nn.Sequential(
            nn.LayerNorm(2 * h),
            nn.Linear(2 * h, 2 * h),
            nn.GELU(),
            nn.Linear(2 * h, tactile_dim),
        )
        self.future_visual_head = nn.Sequential(
            nn.LayerNorm(2 * h),
            nn.Linear(2 * h, 2 * h),
            nn.GELU(),
            nn.Linear(2 * h, visual_dim),
        )
        self.condition_query = nn.Parameter(
            torch.randn(1, self.num_condition_tokens, h) * 0.02
        )
        # history touch, history visual, history state, future touch, future visual
        self.condition_role_embedding = nn.Parameter(torch.randn(1, 5, h) * 0.02)
        self.condition_attn = nn.MultiheadAttention(
            h, heads, dropout=drop_path, batch_first=True
        )
        self.condition_norm = nn.LayerNorm(h)
        self.condition_mlp_norm = nn.LayerNorm(h)
        self.condition_mlp = nn.Sequential(
            nn.Linear(h, 2 * h),
            nn.GELU(),
            nn.Linear(2 * h, h),
        )
        condition_width = self.num_condition_tokens * h
        self.condition_future_tactile_memory = nn.Linear(2 * h, h)
        self.condition_future_visual_memory = nn.Linear(2 * h, h)
        touch_summary_width = 4 * num_hands * tactile_dim
        self.condition_history_head = nn.Sequential(
            nn.LayerNorm(condition_width),
            nn.Linear(condition_width, h),
            nn.GELU(),
            nn.Linear(h, touch_summary_width),
        )
        self.condition_future_tactile_head = nn.Sequential(
            nn.LayerNorm(condition_width),
            nn.Linear(condition_width, h),
            nn.GELU(),
            nn.Linear(h, touch_summary_width),
        )
        # Full pooled-future reconstruction through the 4 query tokens.  Summary
        # statistics alone do not force the tokens to retain the complete future
        # trajectory, especially when all four queries can share information.
        self.condition_future_trajectory_head = nn.Sequential(
            nn.LayerNorm(condition_width),
            nn.Linear(condition_width, h),
            nn.GELU(),
            nn.Linear(h, len(self.future_offsets) * num_hands * tactile_dim),
        )
        self.condition_history_visual_head = nn.Sequential(
            nn.LayerNorm(condition_width),
            nn.Linear(condition_width, h),
            nn.GELU(),
            nn.Linear(h, 2 * visual_dim),
        )
        self.condition_future_visual_head = nn.Sequential(
            nn.LayerNorm(condition_width),
            nn.Linear(condition_width, h),
            nn.GELU(),
            nn.Linear(h, 2 * visual_dim),
        )
        self.condition_history_state_head = nn.Sequential(
            nn.LayerNorm(condition_width),
            nn.Linear(condition_width, h),
            nn.GELU(),
            nn.Linear(h, 2 * state_dim),
        )
        self.apply(self._init)
        nn.init.zeros_(self.future_head[-1].weight)
        nn.init.zeros_(self.future_head[-1].bias)
        nn.init.zeros_(self.future_visual_head[-1].weight)
        nn.init.zeros_(self.future_visual_head[-1].bias)
        for head in (
            self.condition_history_head,
            self.condition_future_tactile_head,
            self.condition_future_trajectory_head,
            self.condition_history_visual_head,
            self.condition_future_visual_head,
            self.condition_history_state_head,
        ):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        visual: torch.Tensor,
        tactile: torch.Tensor,
        *,
        visual_indices: Sequence[int],
        tactile_indices: Sequence[int],
        context_frames: int,
        state_history: torch.Tensor,
        hand_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if tactile.ndim == 4:
            tactile = tactile.unsqueeze(-2)
        if tactile.ndim != 5:
            raise ValueError("tactile context must be [B,T,H,N,D]")
        if visual.ndim == 3:
            visual = visual.unsqueeze(-2)
        batch, tactile_time, hands, _, _ = tactile.shape
        if hands != self.num_hands:
            raise ValueError("unexpected hand count")
        if state_history.shape[:2] != (batch, tactile_time):
            raise ValueError("state history time mismatch")
        if hand_valid is None:
            hand_valid = torch.ones(
                batch, hands, device=tactile.device, dtype=torch.bool
            )
        if hand_valid.shape != (batch, hands):
            raise ValueError(
                f"hand_valid must be [B,{hands}], got {tuple(hand_valid.shape)}"
            )
        hand_valid = hand_valid.to(device=tactile.device, dtype=torch.bool)
        if not hand_valid.any(dim=1).all():
            raise ValueError("each sample must contain at least one valid hand")
        tidx = torch.as_tensor(tactile_indices, device=tactile.device)
        tactile_tokens = (
            self.tactile_in(tactile.flatten(-2))
            + self.hand_embedding[:, None]
            + self.time_embedding[:, tidx][:, :, None, :]
        ).flatten(1, 2)
        tactile_valid = hand_valid[:, None, :].expand(
            batch, tactile_time, hands
        ).flatten(1)
        tactile_tokens = tactile_tokens * tactile_valid[..., None].to(
            tactile_tokens.dtype
        )
        vidx = torch.as_tensor(visual_indices, device=visual.device)
        visual_tokens = (
            self.visual_in(visual)
            + self.time_embedding[:, vidx][:, :, None]
            + self.visual_embedding
        ).flatten(1, 2)
        state_tokens = (
            self.state_in(state_history)
            + self.time_embedding[:, tidx]
            + self.state_embedding
        )
        context = self.context_norm(
            self.context_encoder(
                tactile_tokens,
                src_key_padding_mask=~tactile_valid,
            )
        )
        if self.use_visual:
            fused, _ = self.visual_fusion(
                context, visual_tokens, visual_tokens, need_weights=False
            )
            context = self.visual_fusion_norm(
                context + torch.tanh(self.visual_gate) * fused
            )
        if self.use_state:
            fused, _ = self.state_fusion(
                context, state_tokens, state_tokens, need_weights=False
            )
            context = self.state_fusion_norm(
                context + torch.tanh(self.state_gate) * fused
            )
        pooled = context.mean(dim=1)
        future_indices = torch.as_tensor(
            [context_frames + offset - 1 for offset in self.future_offsets],
            device=context.device,
        )
        queries = (
            self.future_query
            + self.time_embedding[:, future_indices][:, :, None]
            + self.hand_embedding[:, None]
        ).expand(batch, -1, -1, -1)
        count = queries.shape[1]
        query = queries.reshape(batch * count * self.num_hands, 1, -1)
        memory = context[:, None, None].expand(
            batch,
            count,
            self.num_hands,
            context.shape[1],
            -1,
        ).reshape(
            batch * count * self.num_hands,
            context.shape[1],
            -1,
        )
        attended, _ = self.cross_attn(
            query, memory, memory, need_weights=False
        )
        attended = attended.reshape(batch, count, self.num_hands, -1)
        attended = attended * hand_valid[:, None, :, None].to(attended.dtype)
        attended = self.cross_norm(queries + attended)
        pooled = pooled[:, None, None].expand(batch, count, self.num_hands, -1)
        pooled = pooled * hand_valid[:, None, :, None].to(pooled.dtype)
        features = torch.cat((pooled, attended), dim=-1)
        delta = self.future_head(features).reshape(
            batch, count, self.num_hands, self.tactile_dim
        ).unsqueeze(-2)
        delta = delta * hand_valid[:, None, :, None, None].to(delta.dtype)
        current_tactile = tactile[:, -1, :, :1][:, None]
        current_tactile = current_tactile * hand_valid[
            :, None, :, None, None
        ].to(current_tactile.dtype)
        future_tactile = current_tactile + delta

        offset_to_position = {
            offset: position
            for position, offset in enumerate(self.future_offsets)
        }
        visual_positions = [
            offset_to_position[offset]
            for offset in self.visual_future_offsets
        ]
        visual_candidates = features[:, visual_positions]
        visual_weights = hand_valid[:, None, :, None].to(visual_candidates.dtype)
        visual_features = (
            visual_candidates * visual_weights
        ).sum(dim=2) / visual_weights.sum(dim=2).clamp_min(1.0)
        if self.detach_auxiliary_paths:
            visual_features = visual_features.detach()
        visual_delta = self.future_visual_head(visual_features)
        current_visual = visual[:, -1, 0]
        future_visual = current_visual[:, None] + visual_delta

        condition_source = features.detach() if self.detach_auxiliary_paths else features
        future_tactile_source = condition_source * hand_valid[
            :, None, :, None
        ].to(condition_source.dtype)
        future_tactile_memory = self.condition_future_tactile_memory(
            future_tactile_source.flatten(1, 2)
        )
        future_visual_source = condition_source[:, visual_positions]
        future_visual_source = future_visual_source * hand_valid[
            :, None, :, None
        ].to(future_visual_source.dtype)
        future_visual_memory = self.condition_future_visual_memory(
            future_visual_source.flatten(1, 2)
        )
        context_memory = context.detach() if self.detach_auxiliary_paths else context
        context_memory = context_memory * tactile_valid[..., None].to(
            context_memory.dtype
        )
        visual_memory = (
            visual_tokens.detach()
            if self.detach_auxiliary_paths else visual_tokens
        )
        state_memory = (
            state_tokens.detach()
            if self.detach_auxiliary_paths else state_tokens
        )
        role = self.condition_role_embedding
        condition_memory = torch.cat(
            (
                context_memory + role[:, 0],
                visual_memory + role[:, 1],
                state_memory + role[:, 2],
                future_tactile_memory + role[:, 3],
                future_visual_memory + role[:, 4],
            ),
            dim=1,
        )
        condition_key_padding = torch.cat(
            (
                ~tactile_valid,
                torch.zeros(
                    visual_tokens.shape[:2],
                    device=visual_tokens.device,
                    dtype=torch.bool,
                ),
                torch.zeros(
                    state_tokens.shape[:2],
                    device=state_tokens.device,
                    dtype=torch.bool,
                ),
                ~hand_valid[:, None, :].expand(
                    batch, count, hands
                ).flatten(1),
                ~hand_valid[:, None, :].expand(
                    batch, len(visual_positions), hands
                ).flatten(1),
            ),
            dim=1,
        )
        queries = self.condition_query.expand(batch, -1, -1)
        condition, _ = self.condition_attn(
            queries,
            condition_memory,
            condition_memory,
            key_padding_mask=condition_key_padding,
            need_weights=False,
        )
        condition = self.condition_norm(queries + condition)
        condition = condition + self.condition_mlp(
            self.condition_mlp_norm(condition)
        )
        condition_flat = condition.flatten(1)
        history_summary = self.condition_history_head(
            condition_flat
        ).reshape(batch, self.num_hands, 4, self.tactile_dim)
        future_tactile_summary = self.condition_future_tactile_head(
            condition_flat
        ).reshape(batch, self.num_hands, 4, self.tactile_dim)
        future_tactile_trajectory = self.condition_future_trajectory_head(
            condition_flat
        ).reshape(
            batch,
            len(self.future_offsets),
            self.num_hands,
            self.tactile_dim,
        )
        history_visual_summary = self.condition_history_visual_head(
            condition_flat
        ).reshape(batch, 2, self.visual_dim)
        future_visual_summary = self.condition_future_visual_head(
            condition_flat
        ).reshape(batch, 2, self.visual_dim)
        history_state_summary = self.condition_history_state_head(
            condition_flat
        ).reshape(batch, 2, self.state_dim)
        return {
            "future_tactile": future_tactile,
            "future_visual": future_visual,
            "condition_tokens": condition,
            "condition_history_summary": history_summary,
            "condition_future_tactile_summary": future_tactile_summary,
            "condition_future_tactile_trajectory": future_tactile_trajectory,
            "condition_history_visual_summary": history_visual_summary,
            "condition_future_visual_summary": future_visual_summary,
            "condition_history_state_summary": history_state_summary,
        }


__all__ = ["DenseTactilePredictor"]
