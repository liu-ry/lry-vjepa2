"""Shared causal tactile encoder and runtime tensor helpers.

The model definitions are copied from the repository training scripts without
changing their parameters or forward behavior. Stage B imports these helpers
so inference does not need to import the full training runner.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tactile_encoder import TactileEncoder

class CausalTemporalAdapter(nn.Module):
    """Causal temporal attention per spatial token after frame encoding."""

    def __init__(self, dim: int, depth: int, heads: int, max_time: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.time_embedding = nn.Parameter(torch.randn(1, max_time, dim) * 0.02)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, time, spatial, dim = frames.shape
        tokens = frames.permute(0, 2, 1, 3).reshape(batch * spatial, time, dim)
        tokens = tokens + self.time_embedding[:, :time]
        mask = torch.triu(
            torch.ones(time, time, device=frames.device, dtype=torch.bool),
            diagonal=1,
        )
        tokens = self.encoder(tokens, mask=mask)
        return self.norm(tokens).reshape(batch, spatial, time, dim).permute(0, 2, 1, 3)

class ChainTactileEncoder(nn.Module):
    def __init__(
        self,
        *,
        attention_mode: str,
        embed_dim: int,
        depth: int,
        heads: int,
        image_size: tuple[int, int],
        max_time: int,
        causal_depth: int,
        drop_path: float,
    ):
        super().__init__()
        if attention_mode not in ("per_frame_causal", "full_clip_bidirectional"):
            raise ValueError(f"unsupported attention mode {attention_mode!r}")
        self.attention_mode = attention_mode
        self.frame_encoder = TactileEncoder(
            embed_dim=embed_dim,
            in_channels=6,
            patch_size=16,
            tubelet_size=1,
            img_size=image_size,
            num_frames=1,
            depth=depth,
            num_heads=heads,
            use_rope=True,
            drop_path=drop_path,
        )
        self.temporal_adapter = (
            CausalTemporalAdapter(
                embed_dim, causal_depth, heads, max_time, drop_path
            )
            if attention_mode == "per_frame_causal"
            else None
        )

    def forward(self, video: torch.Tensor, token_mask: torch.Tensor | None = None):
        batch, time, channels, height, width = video.shape
        flat = video.reshape(batch * time, channels, height, width)
        flat_mask = token_mask.reshape(batch * time, token_mask.shape[-1]) if token_mask is not None else None
        if self.attention_mode == "per_frame_causal":
            frames = self.frame_encoder(flat, token_mask=flat_mask).squeeze(1)
            frames = frames.reshape(batch, time, frames.shape[-2], frames.shape[-1])
            return self.temporal_adapter(frames)
        tokens = self.frame_encoder(video, token_mask=token_mask)
        return tokens

class FrameQueryTokenPredictor(nn.Module):
    """Causal frame-query predictor producing a full token grid per frame."""

    def __init__(self, dim, hidden, depth, heads, max_time, spatial_tokens, dropout):
        super().__init__()
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.spatial_tokens = int(spatial_tokens)
        self.frame_in = nn.Linear(2 * dim, hidden)
        self.time_embedding = nn.Parameter(torch.randn(1, max_time, hidden) * 0.02)
        self.future_query = nn.Parameter(torch.randn(1, hidden) * 0.02)
        self.spatial_embedding = nn.Parameter(
            torch.randn(1, spatial_tokens, hidden) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=4 * hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.frame_encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.frame_norm = nn.LayerNorm(hidden)
        self.cross_attn = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        tokens,
        memory_mask,
        frame_indices,
        context_indices,
        future_indices,
        base_tokens,
    ):
        batch, frames, spatial, dim = tokens.shape
        if spatial != self.spatial_tokens:
            raise ValueError(
                f"expected {self.spatial_tokens} tokens, got {spatial}"
            )
        valid = memory_mask.bool()
        valid_float = valid.to(tokens.dtype)
        mean = (tokens * valid_float[..., None]).sum(dim=2)
        mean = mean / valid_float.sum(dim=2, keepdim=True).clamp_min(1.0)
        masked = tokens.masked_fill(~valid[..., None], float("-inf"))
        maximum = masked.amax(dim=2)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        frame_features = self.frame_in(torch.cat((mean, maximum), dim=-1))
        frame_features = frame_features + self.time_embedding[
            :, torch.as_tensor(frame_indices, device=tokens.device)
        ]
        frame_valid = valid.any(dim=2)
        memory = self.frame_norm(
            self.frame_encoder(
                frame_features,
                src_key_padding_mask=~frame_valid,
            )
        )
        target_indices = [int(x) for x in context_indices] + [
            int(x) for x in future_indices
        ]
        frame_to_position = {
            int(index): position for position, index in enumerate(frame_indices)
        }
        target_positions = [frame_to_position[index] for index in target_indices]
        target_time = torch.as_tensor(
            target_indices, device=tokens.device, dtype=torch.long
        )
        queries = (
            self.future_query
            + self.time_embedding[:, target_time]
        ).expand(batch, -1, -1)
        query_count = queries.shape[1]
        allowed = torch.arange(frames, device=tokens.device).view(1, 1, frames)
        target_pos = torch.as_tensor(
            target_positions, device=tokens.device
        ).view(1, query_count, 1)
        allowed = allowed <= target_pos
        allowed = allowed.reshape(query_count, frames)
        attended, _ = self.cross_attn(
            queries,
            memory,
            memory,
            attn_mask=~allowed,
            need_weights=False,
        )
        attended = self.cross_norm(queries + attended)
        residual = self.head(
            attended[:, :, None] + self.spatial_embedding
        )
        if base_tokens.ndim == 4:
            output = residual + base_tokens[:, target_positions]
        else:
            output = residual + base_tokens[:, None]
        context_count = len(context_indices)
        return {
            "context_tactile": output[:, :context_count],
            "future_tactile": output[:, context_count:],
        }

def _resize_clip(clip: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if clip.shape[-2:] == size:
        return clip
    batch, time, channels, height, width = clip.shape
    flat = clip.reshape(batch * time, channels, height, width)
    flat = F.interpolate(flat, size=size, mode="bilinear", align_corners=False)
    return flat.reshape(batch, time, channels, *size)

def _causal_delta_clip(clip: torch.Tensor) -> torch.Tensor:
    previous = torch.cat((clip[:, :1], clip[:, :-1]), dim=1)
    return torch.cat((clip, clip - previous), dim=2)

def _pool_chain_tokens(
    tokens: torch.Tensor,
    source_grid: tuple[int, int],
    target_grid: tuple[int, int],
) -> torch.Tensor:
    """Pool ``[B,C,T,N,D]`` latents while preserving each chain and frame."""
    batch, chains, time, count, dim = tokens.shape
    if count == target_grid[0] * target_grid[1]:
        return tokens
    if count != source_grid[0] * source_grid[1]:
        raise ValueError(
            f"source grid {source_grid} does not match {count} tactile tokens"
        )
    flat = tokens.reshape(
        batch * chains * time, count, dim
    ).transpose(1, 2).reshape(
        batch * chains * time,
        dim,
        source_grid[0],
        source_grid[1],
    )
    flat = F.adaptive_avg_pool2d(flat, target_grid)
    return flat.reshape(
        batch * chains * time, dim, target_grid[0] * target_grid[1]
    ).transpose(1, 2).reshape(
        batch, chains, time, target_grid[0] * target_grid[1], dim
    )

def _encode_chains(
    encoder: nn.Module,
    clip: torch.Tensor,
    indices: list[int],
    source_grid: tuple[int, int],
    target_grid: tuple[int, int],
) -> torch.Tensor:
    """Apply one shared encoder to each chain in isolation."""
    batch, chains, _, channels, height, width = clip.shape
    selected = clip[:, :, indices].reshape(
        batch * chains, len(indices), channels, height, width
    )
    tokens = encoder(selected)
    tokens = tokens.reshape(
        batch, chains, len(indices), tokens.shape[-2], tokens.shape[-1]
    )
    return _pool_chain_tokens(tokens, source_grid, target_grid)

def _encode_full_chains(
    encoder: nn.Module,
    clip: torch.Tensor,
    source_grid: tuple[int, int],
    target_grid: tuple[int, int],
) -> torch.Tensor:
    """Encode each complete chain so targets keep their absolute time slots."""
    return _encode_chains(
        encoder,
        clip,
        list(range(clip.shape[2])),
        source_grid,
        target_grid,
    )

def _align_temporal_embedding(
    state: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Load a short-horizon Stage-A encoder into the Stage-B time shell.

    Stage B only encodes the first ``context`` frames.  Absolute positions
    learned by Stage A must therefore stay in their original locations.  The
    trained prefix is copied verbatim; unused future positions are initialized
    from the last trained position.
    """
    key = "temporal_adapter.time_embedding"
    if key not in state or key not in reference:
        return state
    source = state[key]
    target = reference[key]
    if source.shape == target.shape:
        return state
    if source.ndim != 3 or target.ndim != 3 or source.shape[2] != target.shape[2]:
        raise ValueError(
            f"incompatible encoder time embeddings: {source.shape} vs {target.shape}"
        )
    shared = min(source.shape[1], target.shape[1])
    aligned = source.new_zeros(target.shape)
    aligned[:, :shared] = source[:, :shared]
    if target.shape[1] > shared:
        aligned[:, shared:] = source[:, shared - 1 : shared]
    state = dict(state)
    state[key] = aligned
    return state
