"""Autoresearch Stage A for independent tactile chains."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.vjepa_2_1.data import RobotVisionTactileDataset
from app.vjepa_2_1.models.tactile_encoder import TactileEncoder
from app.vjepa_2_1.tactile_training_utils import (
    WarmupCosineMultiplier,
    ensure_writable_output,
    is_main_process,
    setup_distributed,
    unique_output_dir,
    update_ema,
)
from app.vjepa_2_1.train_tactile_two_stage import (
    _auroc,
    _dual_tactile_delta_loss,
    _normalized_context_latent_loss,
    _three_way_group_split,
)


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


class ChainStageAModule(nn.Module):
    def __init__(self, cfg: dict, device: str):
        super().__init__()
        section = cfg["encoder_training"]
        tactile = cfg["tactile"]
        self.cfg = cfg
        self.section = section
        self.device_name = device
        self.image_size = tuple(int(x) for x in tactile.get("img_size", [336, 224]))
        embed_dim = int(tactile.get("embed_dim", 384))
        heads = int(tactile.get("num_heads", 6))
        common = dict(
            embed_dim=embed_dim,
            depth=int(tactile.get("depth", 6)),
            heads=heads,
            image_size=self.image_size,
            max_time=int(cfg["context"]) + int(cfg["horizon"]),
            causal_depth=int(section.get("causal_temporal_depth", 2)),
            drop_path=float(tactile.get("drop_path", 0.1)),
        )
        self.encoder = ChainTactileEncoder(
            attention_mode=str(section.get("temporal_attention", "per_frame_causal")),
            **common,
        ).to(device)
        self.encoder_target = copy.deepcopy(self.encoder).eval()
        for parameter in self.encoder_target.parameters():
            parameter.requires_grad_(False)
        target_grid = tuple(int(x) for x in section.get("latent_spatial_grid", [2, 2]))
        self.target_grid = target_grid
        self.target_tokens = target_grid[0] * target_grid[1]
        self.context_count = int(section.get("latent_context_frames", 4))
        self.objective = str(
            section.get("stage_a_objective", "future_forecast")
        ).lower()
        if self.objective not in (
            "future_forecast",
            "future_sparse",
            "clip_masked",
            "masked_encoding",
        ):
            raise ValueError(
                "unsupported Stage-A objective"
            )
        default_sparse = (1, 2, 4, 8, 16, 30)
        self.sparse_future_offsets = tuple(
            int(x)
            for x in section.get("stage_a_future_offsets", default_sparse)
        )
        self.mask_mode = str(
            section.get("latent_mask_mode", "spatial_tokens")
        ).lower()
        if self.mask_mode not in ("spatial_tokens", "future_frames"):
            raise ValueError(
                "latent_mask_mode must be 'spatial_tokens' or 'future_frames'"
            )
        self.mask_ratio = float(section.get("latent_token_mask_ratio", 0.75))
        self.masked_encoding_probability = float(
            section.get("masked_encoding_probability", 0.5)
        )
        self.predictor = FrameQueryTokenPredictor(
            dim=embed_dim,
            hidden=int(section.get("latent_predictor_dim", embed_dim)),
            depth=int(section.get("latent_predictor_depth", 2)),
            heads=int(section.get("latent_predictor_heads", heads)),
            max_time=int(cfg["context"]) + int(cfg["horizon"]),
            spatial_tokens=self.target_tokens,
            dropout=float(section.get("latent_predictor_drop_path", 0.05)),
        ).to(device)
        self.contact_probe = nn.Sequential(
            nn.LayerNorm(2 * embed_dim), nn.Linear(2 * embed_dim, 1)
        ).to(device)
        self.slip_probe = nn.Sequential(
            nn.LayerNorm(2 * embed_dim), nn.Linear(2 * embed_dim, 1)
        ).to(device)
        self.opt = torch.optim.AdamW(
            list(self.encoder.parameters())
            + list(self.predictor.parameters())
            + list(self.contact_probe.parameters())
            + list(self.slip_probe.parameters()),
            lr=float(section.get("lr", 1e-4)),
            weight_decay=float(section.get("weight_decay", 0.05)),
        )
        self.amp_dtype = (
            torch.bfloat16
            if str(section.get("amp_dtype", "bfloat16"))
            in ("bf16", "bfloat16")
            else torch.float16
        )
        self.use_amp = bool(section.get("amp", True)) and device.startswith("cuda")
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype == torch.float16
        )

    def _make_mask(self, batch: int, frames: int, spatial: int) -> torch.Tensor:
        mask = torch.zeros(
            batch, frames, spatial, device=self.device_name, dtype=torch.bool
        )
        if self.objective == "masked_encoding":
            if torch.rand((), device=self.device_name) >= (
                self.masked_encoding_probability
            ):
                return mask
            mask = (
                torch.rand(
                    batch,
                    frames,
                    spatial,
                    device=self.device_name,
                )
                < self.mask_ratio
            )
            # Keep one visible anchor in every frame so the encoder always
            # receives direct evidence about the frame it is encoding.
            mask[..., 0] = False
            return mask
        if self.objective == "clip_masked":
            height, width = 21, 14
            if spatial != height * width:
                raise ValueError(
                    "clip_masked expects the native 21x14 tactile grid"
                )
            grid = mask.view(batch, frames, height, width)
            for batch_index in range(batch):
                for _ in range(10):
                    spatial_ratio = 0.12 + 0.08 * float(
                        torch.rand((), device=self.device_name)
                    )
                    aspect = 0.75 + 0.75 * float(
                        torch.rand((), device=self.device_name)
                    )
                    area = max(1, round(spatial * spatial_ratio))
                    block_h = min(
                        height, max(1, round(math.sqrt(area * aspect)))
                    )
                    block_w = min(
                        width, max(1, round(math.sqrt(area / aspect)))
                    )
                    temporal_ratio = 0.25 + 0.75 * float(
                        torch.rand((), device=self.device_name)
                    )
                    block_t = min(
                        frames, max(1, round(frames * temporal_ratio))
                    )
                    top = int(
                        torch.randint(
                            0,
                            height - block_h + 1,
                            (),
                            device=self.device_name,
                        )
                    )
                    left = int(
                        torch.randint(
                            0,
                            width - block_w + 1,
                            (),
                            device=self.device_name,
                        )
                    )
                    start = int(
                        torch.randint(
                            0,
                            frames - block_t + 1,
                            (),
                            device=self.device_name,
                        )
                    )
                    grid[
                        batch_index,
                        start : start + block_t,
                        top : top + block_h,
                        left : left + block_w,
                    ] = True
            return mask
        if self.mask_mode == "future_frames":
            mask[:, self.context_count :] = True
        else:
            mask[:, self.context_count :] = (
                torch.rand(
                    batch,
                    frames - self.context_count,
                    spatial,
                    device=self.device_name,
                )
                < self.mask_ratio
            )
            # Keep one visible anchor per future frame so the predictor can
            # estimate the current tactile state while reconstructing the rest.
            mask[:, self.context_count :, 0] = False
        return mask

    def _pool_target(self, tokens: torch.Tensor, valid: torch.Tensor | None = None):
        batch, frames, spatial, dim = tokens.shape
        if (self.target_grid[0] * self.target_grid[1]) == spatial:
            return tokens, valid
        if valid is None:
            valid = torch.ones(
                batch, frames, spatial, device=tokens.device, dtype=torch.bool
            )
        flat_tokens = tokens.reshape(batch * frames, spatial, dim).transpose(1, 2).reshape(
            batch * frames, dim, 21, 14
        )
        flat_valid = valid.reshape(batch * frames, 1, 21, 14).to(tokens.dtype)
        numerator = F.adaptive_avg_pool2d(
            flat_tokens * flat_valid, self.target_grid
        )
        denominator = F.adaptive_avg_pool2d(flat_valid, self.target_grid).clamp_min(1e-8)
        pooled = (numerator / denominator).reshape(
            batch * frames, dim, -1
        ).transpose(1, 2).reshape(batch, frames, -1, dim)
        pooled_valid = (denominator > 0).reshape(batch, frames, -1)
        return pooled, pooled_valid

    def forward(self, batch: dict):
        clip = _resize_clip(batch["tactile"].to(self.device_name), self.image_size)
        clip = _causal_delta_clip(clip)
        batch_size, frames = clip.shape[:2]
        mask = self._make_mask(batch_size, frames, 21 * 14)
        with torch.autocast(
            device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
        ):
            online_fine = self.encoder(clip, token_mask=mask)
            with torch.no_grad():
                target_fine = self.encoder_target(clip)
            online, online_valid = self._pool_target(online_fine, ~mask)
            target, _ = self._pool_target(target_fine)
            if self.objective == "masked_encoding":
                future_indices = []
            elif self.objective == "future_sparse":
                future_indices = [
                    self.context_count + offset - 1
                    for offset in self.sparse_future_offsets
                ]
            else:
                future_indices = list(
                    range(self.context_count, frames)
                )
            prediction = self.predictor(
                online,
                memory_mask=online_valid,
                frame_indices=list(range(frames)),
                context_indices=(
                    list(range(frames))
                    if self.objective == "masked_encoding"
                    else list(range(self.context_count))
                ),
                future_indices=future_indices,
                base_tokens=(
                    online
                    if self.objective == "masked_encoding"
                    else online[:, self.context_count - 1]
                ),
            )
            if self.objective == "masked_encoding":
                future_loss, future_copy, context_cos = (
                    _normalized_context_latent_loss(
                        prediction["context_tactile"].unsqueeze(2),
                        target.unsqueeze(2),
                        scale=float(
                            self.section.get(
                                "latent_context_loss_scale", 0.01
                            )
                        ),
                    )
                )
                context_loss = future_loss
                context_copy = future_copy
                delta_rms = (
                    prediction["context_tactile"] - target
                ).square().mean().sqrt()
            else:
                context_loss, context_copy, context_cos = (
                    _normalized_context_latent_loss(
                        prediction["context_tactile"].unsqueeze(2),
                        target[:, : self.context_count].unsqueeze(2),
                        scale=float(
                            self.section.get(
                                "latent_context_loss_scale", 0.01
                            )
                        ),
                    )
                )
            if self.objective in ("clip_masked", "masked_encoding"):
                prediction_all = torch.cat(
                    (
                        prediction["context_tactile"],
                        prediction["future_tactile"],
                    ),
                    dim=1,
                )
                if self.objective == "clip_masked":
                    future_loss, future_copy, _ = (
                        _normalized_context_latent_loss(
                            prediction_all.unsqueeze(2),
                            target.unsqueeze(2),
                            scale=float(
                                self.section.get(
                                    "latent_clip_loss_scale", 0.01
                                )
                            ),
                        )
                    )
                    delta_rms = (
                        prediction_all - target
                    ).square().mean().sqrt()
            else:
                selected_indices = (
                    future_indices
                    if self.objective == "future_sparse"
                    else list(range(self.context_count, frames))
                )
                future_loss, future_copy, delta_rms = (
                    _dual_tactile_delta_loss(
                        prediction["future_tactile"].unsqueeze(2),
                        target[:, selected_indices].unsqueeze(2),
                        online[:, self.context_count - 1].unsqueeze(1),
                        scale=float(
                            self.section.get("latent_delta_scale", 0.01)
                        ),
                        direction_weight=float(
                            self.section.get("latent_direction_weight", 0.1)
                        ),
                    )
                )
            variance = self._variance(online_fine)
            covariance = self._covariance(online_fine)
            contact_target = batch["contact"].to(self.device_name)[
                :, : self.context_count
            ].float()
            slip_target = batch["slip"].to(self.device_name)[
                :, : self.context_count
            ].float()
            pooled = torch.cat(
                (online_fine.mean(dim=-2), online_fine.amax(dim=-2)), dim=-1
            )[:, : self.context_count]
            contact_logits = self.contact_probe(pooled).squeeze(-1)
            slip_logits = self.slip_probe(pooled).squeeze(-1)
            contact_loss = F.binary_cross_entropy_with_logits(
                contact_logits, contact_target
            )
            slip_loss = F.binary_cross_entropy_with_logits(
                slip_logits, slip_target
            )
            current = F.normalize(pooled[:, 1:], dim=-1)
            previous = F.normalize(pooled[:, :-1], dim=-1)
            feature_change = (1 - (current * previous).sum(-1)).clamp_min(0)
            energy = batch["slip"].to(self.device_name)[
                :, 1 : self.context_count
            ].float()
            target_change = (
                float(self.section.get("temporal_change_scale", 0.05))
                * energy.clamp(0, 2)
            ).detach()
            temporal_loss = F.smooth_l1_loss(feature_change, target_change)
            loss = (
                future_loss
                + float(self.section.get("lambda_context_prediction", 0.1))
                * context_loss
                + float(self.section.get("lambda_variance", 0.05)) * variance
                + float(self.section.get("lambda_covariance", 0.01)) * covariance
                + float(self.section.get("lambda_contact", 0.05)) * contact_loss
                + float(self.section.get("lambda_slip", 0.05)) * slip_loss
                + float(self.section.get("lambda_temporal", 1.0)) * temporal_loss
            )
        eps = 1e-6
        future_rel = (float(future_loss.detach()) + eps) / (
            float(future_copy.detach()) + eps
        )
        context_rel = (float(context_loss.detach()) + eps) / (
            float(context_copy.detach()) + eps
        )
        joint_rel = (0.1 * context_rel + future_rel) / 1.1
        contact_auc = _auroc(contact_logits.detach(), contact_target)
        slip_auc = _auroc(slip_logits.detach(), slip_target)
        metrics = {
            "loss": loss.detach(),
            "future_loss": future_loss.detach(),
            "future_copy": future_copy.detach(),
            "future_relative": torch.tensor(future_rel, device=self.device_name),
            "context_loss": context_loss.detach(),
            "context_copy": context_copy.detach(),
            "context_relative": torch.tensor(context_rel, device=self.device_name),
            "joint_relative": torch.tensor(joint_rel, device=self.device_name),
            "context_cosine": context_cos.detach(),
            "variance": variance.detach(),
            "covariance": covariance.detach(),
            "contact_auroc": torch.tensor(contact_auc, device=self.device_name),
            "slip_auroc": torch.tensor(slip_auc, device=self.device_name),
            "delta_rms": delta_rms.detach(),
        }
        if not self.training:
            future_slice = slice(
                self.context_count
                if self.objective != "masked_encoding"
                else 0,
                None,
            )
            self._last_diagnostics = {
                "predicted_future": (
                    (
                        prediction["future_tactile"]
                        if self.objective != "masked_encoding"
                        else prediction["context_tactile"][:, future_slice]
                    )
                    .detach()
                    .float()
                    .cpu()
                ),
                "target_future": (
                    target[:, future_slice].detach().float().cpu()
                ),
                "base": (
                    online[:, self.context_count - 1].detach().float().cpu()
                ),
                "mask": mask.detach().cpu(),
            }
        return loss, metrics

    @staticmethod
    def _variance(tokens):
        flat = tokens.flatten(0, -2)
        std = torch.sqrt(flat.var(dim=0, unbiased=False) + 1e-4)
        return F.relu(0.1 - std).mean()

    @staticmethod
    def _covariance(tokens):
        flat = tokens.flatten(0, -2).float()
        flat = flat - flat.mean(0, keepdim=True)
        cov = flat.T @ flat / max(1, len(flat) - 1)
        off = cov - torch.diag_embed(torch.diag(cov))
        return off.square().sum() / flat.shape[1]

    def state_dict_for_checkpoint(self):
        unwrapped = self.module if isinstance(self, DistributedDataParallel) else self
        return {
            "encoder": unwrapped.encoder.state_dict(),
            "encoder_target": unwrapped.encoder_target.state_dict(),
            "predictor": unwrapped.predictor.state_dict(),
            "contact_probe": unwrapped.contact_probe.state_dict(),
            "slip_probe": unwrapped.slip_probe.state_dict(),
            "optimizer": unwrapped.opt.state_dict(),
        }


def build_loaders(cfg, *, batch_size, rank, world_size):
    manifest = cfg["manifest"]
    context, horizon = int(cfg["context"]), int(cfg["horizon"])
    min_contact = int(cfg["encoder_min_contact_frames"])
    full = RobotVisionTactileDataset(
        manifest,
        context,
        horizon,
        load_vision=False,
        load_state=False,
        require_contact=True,
        min_contact_frames=min_contact,
    )
    items = [
        item
        for item in full.items
        if len(full.contact_window_starts(item)) > 0
    ]
    train_idx, val_idx, test_idx, split = _three_way_group_split(
        items,
        val_ratio=float(cfg.get("val_ratio", 0.15)),
        test_ratio=float(cfg.get("test_ratio", 0.1)),
        seed=int(cfg.get("split_seed", cfg.get("seed", 0))),
        stable=bool(cfg.get("stable_group_split", True)),
    )
    def make(indices, windows, random_crop):
        return RobotVisionTactileDataset(
            manifest,
            context,
            horizon,
            items=[items[index] for index in indices],
            load_vision=False,
            load_state=False,
            require_contact=True,
            min_contact_frames=min_contact,
            min_hand_contact_frames=1,
            windows_per_episode=windows,
            window_overlap=float(cfg.get("window_overlap", 1.0 / 3.0)),
            random_crop=random_crop,
            num_hands=None,
        )
    train = make(train_idx, int(cfg.get("windows_per_episode", 2)), True)
    val = make(val_idx, int(cfg.get("val_windows_per_episode", 1)), False)
    test = make(test_idx, int(cfg.get("val_windows_per_episode", 1)), False)
    sampler = DistributedSampler(
        train, num_replicas=world_size, rank=rank, shuffle=True
    ) if world_size > 1 else None
    workers = int(cfg.get("num_workers", 2))
    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": True,
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = int(
            cfg.get("prefetch_factor", 2)
        )
        loader_kwargs["persistent_workers"] = bool(
            cfg.get("persistent_workers", True)
        )
    return (
        DataLoader(
            train,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            drop_last=len(train) >= batch_size,
            **loader_kwargs,
        ),
        DataLoader(
            val,
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
        DataLoader(
            test,
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
        split,
    )


def evaluate(module, loader, max_batches=64):
    module.eval()
    totals, count = {}, 0
    with torch.no_grad():
        for batch in loader:
            _, metrics = module(batch)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            count += 1
            if max_batches and count >= max_batches:
                break
    module.train()
    averaged = {
        key: value / max(1, count) for key, value in totals.items()
    }
    if averaged.get("future_copy", 0.0) > 0:
        averaged["future_relative"] = (
            averaged["future_loss"] / averaged["future_copy"]
        )
    if averaged.get("context_copy", 0.0) > 0:
        averaged["context_relative"] = (
            averaged["context_loss"] / averaged["context_copy"]
        )
    averaged["joint_relative"] = (
        0.1 * averaged.get("context_relative", 0.0)
        + averaged.get("future_relative", 0.0)
    ) / 1.1
    return averaged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume-step", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--sharing-strategy",
        choices=("file_system", "file_descriptor"),
        default=None,
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Fixed number of consecutive future frames to forecast.",
    )
    parser.add_argument("--early-min-steps", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument(
        "--masked-encoding-probability",
        type=float,
        default=None,
    )
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--attention", choices=("per_frame_causal", "full_clip_bidirectional"), default=None)
    parser.add_argument("--target-grid", default=None, help="HxW, for example 2x2 or 21x14")
    parser.add_argument(
        "--mask-mode",
        choices=("spatial_tokens", "future_frames"),
        default=None,
        help="Mask a subset of each future frame or every token in future frames",
    )
    parser.add_argument(
        "--objective",
        choices=(
            "future_forecast",
            "future_sparse",
            "clip_masked",
            "masked_encoding",
        ),
        default=None,
        help=(
            "Use fixed-horizon forecasting, whole-clip masking, or "
            "same-frame masked encoding"
        ),
    )
    parser.add_argument(
        "--future-offsets",
        default=None,
        help="Comma-separated offsets for future_sparse, e.g. 1,2,4,8,16,30",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.attention:
        cfg["encoder_training"]["temporal_attention"] = args.attention
    if args.target_grid:
        height, width = args.target_grid.lower().split("x", 1)
        cfg["encoder_training"]["latent_spatial_grid"] = [
            int(height),
            int(width),
        ]
    if args.mask_mode:
        cfg["encoder_training"]["latent_mask_mode"] = args.mask_mode
    if args.objective:
        cfg["encoder_training"]["stage_a_objective"] = args.objective
    if args.future_offsets:
        cfg["encoder_training"]["stage_a_future_offsets"] = [
            int(value) for value in args.future_offsets.split(",")
        ]
    if args.batch_size:
        cfg["encoder_training"]["batch_size"] = int(args.batch_size)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.num_workers is not None:
        cfg["num_workers"] = int(args.num_workers)
    if args.sharing_strategy is not None:
        cfg["tensor_sharing_strategy"] = args.sharing_strategy
    if args.horizon is not None:
        if args.horizon <= 0:
            parser.error("--horizon must be positive")
        cfg["horizon"] = int(args.horizon)
    if args.early_min_steps is not None:
        cfg["encoder_training"]["early_stopping_min_steps"] = int(
            args.early_min_steps
        )
    if args.max_test_batches is not None:
        cfg["encoder_training"]["max_test_batches"] = int(
            args.max_test_batches
        )
    if args.masked_encoding_probability is not None:
        if not 0.0 <= args.masked_encoding_probability <= 1.0:
            parser.error(
                "--masked-encoding-probability must be in [0, 1]"
            )
        cfg["encoder_training"]["masked_encoding_probability"] = float(
            args.masked_encoding_probability
        )
    if args.patience is not None:
        cfg["encoder_training"]["early_stopping_patience"] = int(
            args.patience
        )
    rank, world_size, device = setup_distributed()
    random.seed(int(cfg.get("seed", 0)) + rank)
    torch.manual_seed(int(cfg.get("seed", 0)) + rank)
    torch.multiprocessing.set_sharing_strategy(
        str(cfg.get("tensor_sharing_strategy", "file_descriptor"))
    )
    output = Path(ensure_writable_output(unique_output_dir(args.output))) if rank == 0 else Path(args.output)
    if world_size > 1:
        obj = [str(output) if rank == 0 else None]
        dist.broadcast_object_list(obj, src=0)
        output = Path(obj[0])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "resolved_config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
        )
    if world_size > 1:
        dist.barrier()
    module = ChainStageAModule(cfg, device)
    if world_size > 1:
        module = DistributedDataParallel(module, device_ids=[torch.cuda.current_device()], find_unused_parameters=False)
    base = module.module if world_size > 1 else module
    start_step = 0
    best = math.inf
    best_step = None
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        base.encoder.load_state_dict(state["encoder"])
        base.encoder_target.load_state_dict(
            state.get("encoder_target", state["encoder"])
        )
        base.predictor.load_state_dict(state["predictor"])
        base.contact_probe.load_state_dict(state["contact_probe"])
        base.slip_probe.load_state_dict(state["slip_probe"])
        if state.get("optimizer") is not None:
            base.opt.load_state_dict(state["optimizer"])
        start_step = int(
            args.resume_step
            if args.resume_step is not None
            else state.get("step", 0)
        )
        best = float(state.get("score", best))
        best_step = start_step if math.isfinite(best) else None
    train_loader, val_loader, test_loader, split = build_loaders(
        cfg, batch_size=int(cfg["encoder_training"].get("batch_size", 2)), rank=rank, world_size=world_size
    )
    section = cfg["encoder_training"]
    configured_steps = (
        args.max_steps
        if args.max_steps is not None
        else section.get("max_steps")
    )
    steps = (
        None
        if configured_steps is None or int(configured_steps) <= 0
        else int(configured_steps)
    )
    min_steps = int(section.get("early_stopping_min_steps", 3000))
    patience = int(section.get("early_stopping_patience", 8))
    min_delta = float(section.get("early_stopping_min_delta", 0.0))
    check_every = int(section.get("checkpoint_every_steps", 250))
    schedule_steps = int(
        section.get(
            "lr_schedule_steps",
            steps
            if steps is not None
            else max(8000, min_steps + patience * check_every),
        )
    )
    scheduler = WarmupCosineMultiplier(
        base.opt,
        schedule_steps,
        int(section.get("warmup_steps", 200)),
        min_lr_scale=float(section.get("min_lr_scale", 0.02)),
    )
    scheduler.jump_to(start_step)
    bad = 0
    records = {}
    step = start_step
    while steps is None or step < steps:
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(step)
        for batch in train_loader:
            step += 1
            base.opt.zero_grad(set_to_none=True)
            loss, metrics = module(batch)
            base.scaler.scale(loss).backward()
            base.scaler.unscale_(base.opt)
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            base.scaler.step(base.opt)
            base.scaler.update()
            scheduler.step()
            update_ema(base.encoder, base.encoder_target, float(section.get("ema_encoder", 0.999)))
            if is_main_process() and (step == 1 or step % int(section.get("log_every", 20)) == 0):
                print(f"[stageA] step={step} " + " ".join(f"{k}={float(v):.5f}" for k,v in metrics.items()), flush=True)
            if step % check_every == 0 or (
                steps is not None and step >= steps
            ):
                val = evaluate(module, val_loader, int(section.get("max_val_batches", 64)))
                score = val["joint_relative"] + float(section.get("probe_selection_weight", 0.001)) * (2 - val["contact_auroc"] - val["slip_auroc"])
                if is_main_process():
                    payload = {
                        **base.state_dict_for_checkpoint(),
                        "step": step,
                        "score": score,
                    }
                    torch.save(payload, output / "latest.pt")
                    if score < best:
                        improvement = best - score
                        best, best_step, bad = score, step, 0
                        torch.save(payload, output / "best.pt")
                    else:
                        improvement = 0.0
                    if step >= min_steps and improvement < min_delta:
                        bad += 1
                    print(f"[stageA] val step={step} score={score:.6f} best={best:.6f}@{best_step} bad={bad}/{patience}", flush=True)
                    records[str(step)] = {
                        "score": score,
                        "validation": val,
                        "best_step": best_step,
                        "bad_checks": bad,
                        "improvement": improvement,
                    }
                    (output / "metrics.json").write_text(
                        json.dumps(
                            {
                                "best_step": best_step,
                                "best_score": best,
                                "records": records,
                            },
                            indent=2,
                            sort_keys=True,
                        ),
                        encoding="utf-8",
                    )
                stop = step >= min_steps and bad >= patience
                if world_size > 1:
                    stop_tensor = torch.tensor([int(stop)], device=device)
                    dist.broadcast(stop_tensor, src=0)
                    stop = bool(stop_tensor.item())
                if stop:
                    if is_main_process():
                        print(f"[stageA] early stop {step}", flush=True)
                    break
            if steps is not None and step >= steps:
                break
        else:
            continue
        break
    if world_size > 1:
        dist.barrier()
    if is_main_process():
        best_payload = torch.load(
            output / "best.pt", map_location=device, weights_only=False
        )
        base.encoder.load_state_dict(best_payload["encoder"])
        base.encoder_target.load_state_dict(
            best_payload.get("encoder_target", best_payload["encoder"])
        )
        base.predictor.load_state_dict(best_payload["predictor"])
        base.contact_probe.load_state_dict(best_payload["contact_probe"])
        base.slip_probe.load_state_dict(best_payload["slip_probe"])
        test = evaluate(module, test_loader, int(section.get("max_test_batches", 0)))
        (output / "test_metrics.json").write_text(json.dumps({"test_metrics":test,"split":split}, indent=2), encoding="utf-8")
        print(f"[stageA] finished best={best_step}@{best}", flush=True)


if __name__ == "__main__":
    main()
