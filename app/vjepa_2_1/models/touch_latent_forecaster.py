"""Masked context/future tactile-latent predictor used by Stage A."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class MaskedTouchLatentPredictor(nn.Module):
    """Joint context reconstruction and masked future-token prediction.

    The encoder receives a token mask before its Transformer blocks.  This
    predictor then consumes the visible coarse spatial grid and predicts every
    requested context/future token.  Context and prediction losses use separate
    output heads but share the temporal-spatial predictor backbone.
    """

    def __init__(
        self,
        tactile_dim: int = 384,
        hidden_dim: int = 384,
        depth: int = 2,
        heads: int = 6,
        max_time: int = 34,
        num_hands: int = 2,
        spatial_tokens: int = 16,
        query_chunk_size: int = 4,
        drop_path: float = 0.05,
    ):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if spatial_tokens <= 0:
            raise ValueError("spatial_tokens must be positive")
        self.tactile_dim = int(tactile_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_hands = int(num_hands)
        self.spatial_tokens = int(spatial_tokens)
        self.query_chunk_size = int(query_chunk_size)
        if self.query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        h = self.hidden_dim
        self.input_proj = nn.Linear(self.tactile_dim, h)
        self.hand_embedding = nn.Parameter(torch.randn(1, num_hands, h) * 0.02)
        self.time_embedding = nn.Parameter(torch.randn(1, int(max_time), h) * 0.02)
        self.spatial_embedding = nn.Parameter(
            torch.randn(1, self.spatial_tokens, h) * 0.02
        )
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
        self.future_query = nn.Parameter(torch.randn(1, h) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            h, heads, dropout=drop_path, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(h)
        self.context_head = nn.Sequential(
            nn.LayerNorm(2 * h),
            nn.Linear(2 * h, 2 * h),
            nn.GELU(),
            nn.Linear(2 * h, self.tactile_dim),
        )
        self.future_head = nn.Sequential(
            nn.LayerNorm(2 * h),
            nn.Linear(2 * h, 2 * h),
            nn.GELU(),
            nn.Linear(2 * h, self.tactile_dim),
        )
        self.apply(self._init)
        nn.init.zeros_(self.context_head[-1].weight)
        nn.init.zeros_(self.context_head[-1].bias)
        nn.init.zeros_(self.future_head[-1].weight)
        nn.init.zeros_(self.future_head[-1].bias)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        tactile_tokens: torch.Tensor,
        *,
        memory_mask: torch.Tensor,
        frame_indices: Sequence[int],
        context_indices: Sequence[int],
        future_indices: Sequence[int],
        base_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict context and future coarse tactile latents.

        Args:
            tactile_tokens: ``[B,F,H,N,D]`` online encoder output.
            memory_mask: ``[B,F,H,N]`` boolean, true where the token is visible.
            frame_indices: source frame index for each ``F`` dimension entry.
            context_indices: source frame indices supervised by context loss.
            future_indices: source frame indices supervised by prediction loss.
            base_tokens: ``[B,H,N,D]`` current latent used by residual output.
        """
        if tactile_tokens.ndim != 5:
            raise ValueError("tactile_tokens must be [B,F,H,N,D]")
        batch, frames, hands, spatial, dim = tactile_tokens.shape
        if (
            hands != self.num_hands
            or spatial != self.spatial_tokens
            or dim != self.tactile_dim
        ):
            raise ValueError("unexpected tactile token shape")
        if memory_mask.shape != (batch, frames, hands, spatial):
            raise ValueError("memory_mask shape mismatch")
        if base_tokens.shape != (batch, hands, spatial, dim):
            raise ValueError("base_tokens shape mismatch")
        if len(frame_indices) != frames:
            raise ValueError("frame_indices length mismatch")

        visible = memory_mask.to(device=tactile_tokens.device, dtype=torch.bool)
        if not visible.flatten(1).any(dim=1).all():
            raise ValueError("each sample must retain at least one visible token")
        fidx = torch.as_tensor(frame_indices, device=tactile_tokens.device)
        tokens = (
            self.input_proj(tactile_tokens)
            + self.time_embedding[:, fidx][:, :, None, None, :]
            + self.hand_embedding[:, None, :, None, :]
            + self.spatial_embedding[:, None, None, :, :]
        ).flatten(1, 3)
        flat_visible = visible.flatten(1)
        memory = self.context_encoder(
            tokens,
            src_key_padding_mask=~flat_visible,
        )
        memory = self.context_norm(memory)
        visible_float = flat_visible.to(dtype=memory.dtype)
        pooled_context = (
            memory * visible_float[..., None]
        ).sum(dim=1) / visible_float.sum(dim=1, keepdim=True).clamp_min(1.0)

        target_indices = [int(x) for x in context_indices] + [
            int(x) for x in future_indices
        ]
        if not target_indices:
            raise ValueError("at least one context or future target is required")
        index_to_frame = {int(index): position for position, index in enumerate(frame_indices)}
        missing = [index for index in target_indices if index not in index_to_frame]
        if missing:
            raise ValueError(f"target frame indices are not encoded: {missing}")
        target_positions = [index_to_frame[index] for index in target_indices]
        target_time = self.time_embedding[:, torch.as_tensor(
            [frame_indices[position] for position in target_positions],
            device=tactile_tokens.device,
        )]
        queries = (
            self.future_query
            + target_time[:, :, None, None, :]
            + self.hand_embedding[:, None, :, None, :]
            + self.spatial_embedding[:, None, None, :, :]
        ).expand(batch, -1, -1, -1, -1)
        count = queries.shape[1]
        attended_chunks = []
        for start in range(0, count, self.query_chunk_size):
            end = min(count, start + self.query_chunk_size)
            chunk = end - start
            flat_queries = queries[:, start:end].reshape(
                batch * chunk * hands * spatial, 1, -1
            )
            expanded_memory = memory[:, None, None, None].expand(
                batch,
                chunk,
                hands,
                spatial,
                frames * hands * spatial,
                -1,
            ).reshape(
                batch * chunk * hands * spatial, -1, self.hidden_dim
            )
            key_padding = (~flat_visible)[:, None, None, None].expand(
                batch, chunk, hands, spatial, frames * hands * spatial
            ).reshape(batch * chunk * hands * spatial, -1)
            chunk_attended, _ = self.cross_attn(
                flat_queries,
                expanded_memory,
                expanded_memory,
                key_padding_mask=key_padding,
                need_weights=False,
            )
            attended_chunks.append(
                chunk_attended.reshape(batch, chunk, hands, spatial, -1)
            )
        attended = torch.cat(attended_chunks, dim=1)
        attended = self.cross_norm(queries + attended)
        pooled = pooled_context[:, None, None, None].expand(
            batch, count, hands, spatial, -1
        )
        features = torch.cat((pooled, attended), dim=-1)
        context_count = len(context_indices)
        context_delta = self.context_head(
            features[:, :context_count]
        ).reshape(batch, context_count, hands, spatial, dim)
        future_delta = self.future_head(
            features[:, context_count:]
        ).reshape(batch, count - context_count, hands, spatial, dim)
        base = base_tokens[:, None]
        return {
            "context_tactile": base + context_delta,
            "future_tactile": base + future_delta,
            "context_delta": context_delta,
            "future_delta": future_delta,
        }


__all__ = ["MaskedTouchLatentPredictor"]
