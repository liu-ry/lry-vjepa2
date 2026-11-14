"""Four-chain tactile world predictor with two visual/state arm streams.

The tactile encoder stays chain-local in Stage B.  This module is the first
place where the four tactile chains and the two visual/state arms are fused.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class ChainWorldPredictor(nn.Module):
    def __init__(
        self,
        tactile_dim: int,
        visual_dim: int,
        state_dim: int,
        spatial_tokens: int,
        hidden_dim: int = 384,
        depth: int = 2,
        condition_depth: int = 2,
        heads: int = 6,
        max_time: int = 40,
        num_chains: int = 4,
        num_arms: int = 2,
        future_offsets: Sequence[int] = tuple(range(1, 31)),
        visual_future_offsets: Sequence[int] = tuple(range(2, 31, 2)),
        num_condition_tokens: int = 4,
        query_future_segments: Sequence[Sequence[int]] | None = None,
        projection_type: str = "linear",
        drop_path: float = 0.05,
    ):
        super().__init__()
        self.tactile_dim = int(tactile_dim)
        self.visual_dim = int(visual_dim)
        self.state_dim = int(state_dim)
        self.spatial_tokens = int(spatial_tokens)
        self.hidden_dim = int(hidden_dim)
        self.num_chains = int(num_chains)
        self.num_arms = int(num_arms)
        self.future_offsets = tuple(int(x) for x in future_offsets)
        self.visual_future_offsets = tuple(
            int(x) for x in visual_future_offsets
        )
        self.num_condition_tokens = int(num_condition_tokens)
        self.projection_type = str(projection_type).lower()
        if self.projection_type not in ("linear", "mlp", "normed_mlp"):
            raise ValueError(
                "projection_type must be 'linear', 'mlp', or 'normed_mlp'"
            )
        if self.num_condition_tokens <= 0:
            raise ValueError("num_condition_tokens must be positive")
        if query_future_segments is None:
            query_future_segments = self._balanced_segments(
                len(self.future_offsets), self.num_condition_tokens
            )
        self.query_future_segments = tuple(
            tuple(int(i) for i in segment)
            for segment in query_future_segments
        )
        if len(self.query_future_segments) != self.num_condition_tokens:
            raise ValueError("one future segment is required per query token")
        flattened_segments = sorted(
            index
            for segment in self.query_future_segments
            for index in segment
        )
        if flattened_segments != list(range(len(self.future_offsets))):
            raise ValueError(
                "query_future_segments must partition every future offset once"
            )

        h = self.hidden_dim
        def projection(width: int) -> nn.Module:
            if self.projection_type == "linear":
                return nn.Linear(width, h)
            layers: list[nn.Module] = []
            if self.projection_type == "normed_mlp":
                layers.append(nn.LayerNorm(width))
            layers.extend((
                nn.Linear(width, h),
                nn.GELU(),
                nn.Linear(h, h),
            ))
            return nn.Sequential(*layers)

        self.tactile_in = projection(tactile_dim)
        self.visual_in = projection(visual_dim)
        self.state_in = projection(state_dim)
        self.chain_embedding = nn.Parameter(torch.randn(1, num_chains, h) * 0.02)
        self.arm_embedding = nn.Parameter(torch.randn(1, num_arms, h) * 0.02)
        self.time_embedding = nn.Parameter(torch.randn(1, max_time, h) * 0.02)
        self.spatial_embedding = nn.Parameter(
            torch.randn(1, self.spatial_tokens, h) * 0.02
        )
        self.modality_embedding = nn.Parameter(torch.randn(1, 3, h) * 0.02)
        self.visual_gate = nn.Parameter(torch.full((1,), 0.05))
        self.state_gate = nn.Parameter(torch.full((1,), 0.05))

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
            nn.LayerNorm(h),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, tactile_dim),
        )
        self.future_visual_head = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, visual_dim),
        )

        self.condition_query = nn.Parameter(
            torch.randn(1, self.num_condition_tokens, h) * 0.02
        )
        self.condition_role = nn.Parameter(
            torch.randn(1, self.num_condition_tokens, h) * 0.02
        )
        self.condition_attn = nn.MultiheadAttention(
            h, heads, dropout=drop_path, batch_first=True
        )
        condition_layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=heads,
            dim_feedforward=2 * h,
            dropout=drop_path,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.condition_encoder = nn.TransformerEncoder(
            condition_layer, num_layers=condition_depth
        )
        self.condition_norm = nn.LayerNorm(h)

        summary_width = 4 * self.num_chains * self.tactile_dim
        visual_summary_width = 2 * self.num_arms * self.visual_dim
        state_summary_width = 2 * self.num_arms * self.state_dim
        self.history_summary_head = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, summary_width),
        )
        self.future_summary_head = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, summary_width),
        )
        self.future_segment_head = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(h),
                    nn.Linear(h, h),
                    nn.GELU(),
                    nn.Linear(h, len(segment) * self.num_chains * tactile_dim),
                )
                for segment in self.query_future_segments
            ]
        )
        self.future_visual_summary_head = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, visual_summary_width),
        )
        self.history_state_summary_head = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, state_summary_width),
        )
        for head in (
            self.history_summary_head,
            self.future_summary_head,
            self.future_visual_summary_head,
            self.history_state_summary_head,
        ):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        for head in self.future_segment_head:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        nn.init.zeros_(self.future_head[-1].weight)
        nn.init.zeros_(self.future_head[-1].bias)
        nn.init.zeros_(self.future_visual_head[-1].weight)
        nn.init.zeros_(self.future_visual_head[-1].bias)

    @staticmethod
    def _balanced_segments(count: int, groups: int) -> list[tuple[int, ...]]:
        boundaries = [
            round(index * count / groups) for index in range(groups + 1)
        ]
        return [
            tuple(range(boundaries[index], boundaries[index + 1]))
            for index in range(groups)
        ]

    @staticmethod
    def _summary(tokens: torch.Tensor) -> torch.Tensor:
        """Summarize ``[B,T,C,N,D]`` with four time/spatial statistics."""
        spatial_mean = tokens.mean(dim=-2)
        spatial_max = tokens.amax(dim=-2)
        return torch.stack(
            (
                spatial_mean.mean(dim=1),
                spatial_mean.amax(dim=1),
                spatial_max.mean(dim=1),
                spatial_max.amax(dim=1),
            ),
            dim=2,
        )

    @staticmethod
    def _temporal_summary(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("temporal summaries require [B,T,D]")
        return torch.stack((tokens.mean(dim=1), tokens.amax(dim=1)), dim=1)

    def forward(
        self,
        tactile: torch.Tensor,
        tactile_indices: Sequence[int],
        chain_valid: torch.Tensor,
        visual: torch.Tensor,
        visual_indices: Sequence[int],
        arm_valid: torch.Tensor,
        state_history: torch.Tensor,
        context_frames: int,
    ) -> dict[str, torch.Tensor]:
        batch, chains, time, spatial, dim = tactile.shape
        if chains != self.num_chains or spatial != self.spatial_tokens:
            raise ValueError("unexpected tactile chain shape")
        if dim != self.tactile_dim:
            raise ValueError("tactile embedding width does not match predictor")
        chain_valid = chain_valid.to(device=tactile.device, dtype=torch.bool)
        arm_valid = arm_valid.to(device=tactile.device, dtype=torch.bool)
        if chain_valid.shape != (batch, chains):
            raise ValueError("chain_valid must be [B,4]")
        if arm_valid.shape != (batch, self.num_arms):
            raise ValueError("arm_valid must be [B,2]")
        if not chain_valid.any(dim=1).all():
            raise ValueError("every batch item must contain a valid tactile chain")

        tactile_time = torch.as_tensor(tactile_indices, device=tactile.device)
        tactile_tokens = (
            self.tactile_in(tactile)
            + self.chain_embedding[:, :, None, None]
            + self.time_embedding[:, tactile_time][:, None, :, None]
            + self.spatial_embedding[:, None, None]
            + self.modality_embedding[:, 0, None, None, None]
        )
        tactile_mask = chain_valid[:, :, None, None].expand(
            batch, chains, time, spatial
        )
        tactile_tokens = tactile_tokens.flatten(1, 3)
        tactile_mask = tactile_mask.flatten(1)
        tactile_tokens = tactile_tokens * tactile_mask[..., None].to(
            tactile_tokens.dtype
        )
        context = self.context_norm(
            self.context_encoder(
                tactile_tokens, src_key_padding_mask=~tactile_mask
            )
        )

        visual_time = torch.as_tensor(visual_indices, device=visual.device)
        visual_tokens = (
            self.visual_in(visual)
            + self.arm_embedding[:, :, None]
            + self.time_embedding[:, visual_time][:, None]
            + self.modality_embedding[:, 1, None, None]
        )
        visual_mask = arm_valid[:, :, None].expand(
            batch, self.num_arms, visual_tokens.shape[2]
        )
        visual_tokens = visual_tokens.flatten(1, 2)
        visual_mask = visual_mask.flatten(1)
        visual_tokens = visual_tokens * visual_mask[..., None].to(
            visual_tokens.dtype
        )
        visual_context = self.visual_fusion(
            context,
            visual_tokens,
            visual_tokens,
            key_padding_mask=~visual_mask,
            need_weights=False,
        )[0]
        context = self.visual_fusion_norm(
            context + torch.tanh(self.visual_gate) * visual_context
        )

        if state_history.shape[:2] != (batch, self.num_arms):
            raise ValueError("state_history must be [B,2,T,D]")
        if state_history.shape[-2] != time:
            raise ValueError("state and tactile context lengths differ")
        state_tokens = (
            self.state_in(state_history)
            + self.arm_embedding[:, :, None]
            + self.time_embedding[:, tactile_time][:, None]
            + self.modality_embedding[:, 2, None, None]
        )
        state_mask = arm_valid[:, :, None].expand(
            batch, self.num_arms, time
        )
        state_tokens = state_tokens.flatten(1, 2)
        state_mask = state_mask.flatten(1)
        state_context = self.state_fusion(
            context,
            state_tokens,
            state_tokens,
            key_padding_mask=~state_mask,
            need_weights=False,
        )[0]
        context = self.state_fusion_norm(
            context + torch.tanh(self.state_gate) * state_context
        )

        future_time = torch.as_tensor(
            [context_frames + offset - 1 for offset in self.future_offsets],
            device=tactile.device,
        )
        tactile_queries = (
            self.future_query
            + self.chain_embedding[:, :, None]
            + self.time_embedding[:, future_time][:, None]
        ).expand(batch, -1, -1, -1)
        flat_queries = tactile_queries.reshape(
            batch, chains * len(future_time), self.hidden_dim
        )
        attended = self.cross_attn(
            flat_queries, context, context, need_weights=False
        )[0]
        attended = self.cross_norm(
            attended
            + flat_queries
        ).reshape(batch, chains, len(future_time), self.hidden_dim)
        future_tactile = tactile[:, :, -1, None] + self.future_head(
            attended[:, :, :, None] + self.spatial_embedding
        )
        future_tactile = future_tactile * chain_valid[
            :, :, None, None, None
        ].to(future_tactile.dtype)

        visual_future_time = torch.as_tensor(
            [
                context_frames + offset - 1
                for offset in self.visual_future_offsets
            ],
            device=tactile.device,
        )
        visual_queries = (
            self.future_query
            + self.arm_embedding[:, :, None]
            + self.time_embedding[:, visual_future_time][:, None]
        ).expand(batch, -1, -1, -1)
        flat_visual_queries = visual_queries.reshape(
            batch,
            self.num_arms * len(visual_future_time),
            self.hidden_dim,
        )
        visual_attended = self.cross_attn(
            flat_visual_queries, context, context, need_weights=False
        )[0]
        visual_attended = self.cross_norm(
            visual_attended + flat_visual_queries
        )
        future_visual = visual[:, :, -1, None] + self.future_visual_head(
            visual_attended
        ).reshape(
            batch,
            self.num_arms,
            len(visual_future_time),
            self.visual_dim,
        )
        future_visual = future_visual * arm_valid[:, :, None, None].to(
            future_visual.dtype
        )

        history_chain_tokens = tactile_tokens.reshape(
            batch, chains, time, spatial, self.hidden_dim
        )[:, :, :, 0]
        future_chain_tokens = (
            self.tactile_in(future_tactile.mean(dim=-2))
            + self.chain_embedding[:, :, None]
            + self.modality_embedding[:, 0, None, None]
        )
        future_visual_summary = (
            self.visual_in(future_visual.mean(dim=2))
            + self.arm_embedding
            + self.modality_embedding[:, 1, None]
        )
        condition_memory = torch.cat(
            (
                history_chain_tokens.flatten(1, 2),
                future_chain_tokens.flatten(1, 2),
                visual_tokens,
                state_tokens,
                future_visual_summary,
            ),
            dim=1,
        )
        future_chain_mask = chain_valid[:, :, None].expand(
            batch, chains, len(future_time)
        ).flatten(1)
        condition_padding = torch.cat(
            (
                ~tactile_mask.reshape(batch, chains, time, spatial)[:, :, :, 0]
                .flatten(1),
                ~future_chain_mask,
                ~visual_mask,
                ~state_mask,
                ~arm_valid,
            ),
            dim=1,
        )
        condition_memory = condition_memory * (~condition_padding)[..., None].to(
            condition_memory.dtype
        )
        condition = self.condition_query.expand(batch, -1, -1)
        condition_initial = condition + self.condition_role
        attended_condition = self.condition_attn(
            condition_initial,
            condition_memory,
            condition_memory,
            key_padding_mask=condition_padding,
            need_weights=False,
        )[0]
        condition = self.condition_norm(
            condition_initial + attended_condition
        )
        condition = self.condition_encoder(condition)

        history_summary = self.history_summary_head(condition).reshape(
            batch,
            self.num_condition_tokens,
            chains,
            4,
            self.tactile_dim,
        )
        future_summary = self.future_summary_head(condition).reshape(
            batch,
            self.num_condition_tokens,
            chains,
            4,
            self.tactile_dim,
        )
        future_segments = [
            head(condition[:, index])
            .reshape(batch, len(segment), chains, self.tactile_dim)
            for index, (head, segment) in enumerate(
                zip(self.future_segment_head, self.query_future_segments)
            )
        ]
        future_token_trajectory = torch.cat(
            [
                segment
                for _, segment in sorted(
                    zip(self.query_future_segments, future_segments),
                    key=lambda pair: pair[0][0] if pair[0] else len(self.future_offsets),
                )
            ],
            dim=1,
        )
        future_visual_summary_prediction = self.future_visual_summary_head(
            condition
        ).reshape(
            batch,
            self.num_condition_tokens,
            self.num_arms,
            2,
            self.visual_dim,
        )
        history_state_summary = self.history_state_summary_head(
            condition
        ).reshape(
            batch,
            self.num_condition_tokens,
            self.num_arms,
            2,
            self.state_dim,
        )
        return {
            "future_tactile": future_tactile,
            "future_visual": future_visual,
            "condition_tokens": condition,
            "condition_history_summary": history_summary,
            "condition_future_summary": future_summary,
            "condition_future_trajectory": future_token_trajectory,
            "condition_future_visual_summary": future_visual_summary_prediction,
            "condition_history_state_summary": history_state_summary,
        }


__all__ = ["ChainWorldPredictor"]
