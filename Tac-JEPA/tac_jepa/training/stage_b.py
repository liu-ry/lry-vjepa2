"""Stage-B future tactile latent predictor training."""

from __future__ import annotations

import argparse
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

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tac_jepa.data.chain import ChainEpisodeDataset
from tac_jepa.models.chain_encoder import (
    ChainTactileEncoder,
    _align_temporal_embedding,
    _causal_delta_clip,
    _encode_chains,
    _encode_full_chains,
    _resize_clip,
)
from tac_jepa.models.predictor import ChainWorldPredictor
from tac_jepa.training.objectives import (
    _auroc,
    _dual_tactile_delta_loss,
    _normalized_summary_loss,
    _summary_4way,
    _temporal_2way,
    _token_diversity_loss,
)
from tac_jepa.training.runtime import (
    WarmupCosineMultiplier,
    ensure_writable_output,
    is_main_process,
    setup_distributed,
    unique_output_dir,
)
from tac_jepa.training.splits import _three_way_group_split
from tac_jepa.vision.builder import build_visual as _build_visual
from tac_jepa.vision.ops import (
    _prepare_visual_gpu,
    _run_visual,
    _visual_tokens,
)


def _load_config(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("manifest", "context", "horizon", "tactile", "predictor_training"):
        if key not in cfg:
            raise ValueError(f"config is missing {key!r}")
    return cfg

def _average(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = metrics[0]
    return {
        key: sum(item[key] for item in metrics) / len(metrics)
        for key in keys
    }

def _build_loaders(cfg: dict, *, batch_size: int, rank: int, world_size: int):
    context, horizon = int(cfg["context"]), int(cfg["horizon"])
    full = ChainEpisodeDataset(
        cfg["manifest"],
        context=context,
        horizon=horizon,
        require_contact=True,
        min_contact_frames=int(cfg.get("min_contact_frames", 4)),
        min_chain_contact_frames=int(cfg.get("min_chain_contact_frames", 1)),
        load_vision=True,
        load_state=True,
        vision_size=int(cfg.get("crop_size", 384)),
        normalize_vision=bool(cfg.get("normalize_vision", True)),
        defer_image_conversion=True,
        random_crop=False,
        balance_single_dual=bool(cfg.get("balance_single_dual", True)),
        dual_window_probability=float(
            cfg.get("dual_window_probability", 0.5)
        ),
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
        return ChainEpisodeDataset(
            cfg["manifest"],
            context=context,
            horizon=horizon,
            items=[items[index] for index in indices],
            windows_per_episode=windows,
            window_overlap=float(cfg.get("window_overlap", 1.0 / 3.0)),
            random_crop=random_crop,
            balance_single_dual=bool(
                cfg.get("balance_single_dual", True)
            ),
            dual_window_probability=float(
                cfg.get("dual_window_probability", 0.5)
            ),
            require_contact=True,
            min_contact_frames=int(cfg.get("min_contact_frames", 4)),
            min_chain_contact_frames=int(
                cfg.get("min_chain_contact_frames", 1)
            ),
            load_vision=True,
            load_state=True,
            vision_size=int(cfg.get("crop_size", 384)),
            normalize_vision=bool(cfg.get("normalize_vision", True)),
            defer_image_conversion=True,
        )

    train = make(
        train_idx, int(cfg.get("windows_per_episode", 2)), True
    )
    val = make(
        val_idx, int(cfg.get("val_windows_per_episode", 1)), False
    )
    test = make(
        test_idx, int(cfg.get("val_windows_per_episode", 1)), False
    )
    sampler = (
        DistributedSampler(
            train,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(cfg.get("seed", 0)),
            drop_last=True,
        )
        if world_size > 1
        else None
    )
    workers = int(cfg.get("num_workers", 2))
    kwargs = {}
    if workers > 0:
        kwargs.update(
            prefetch_factor=int(cfg.get("prefetch_factor", 2)),
            persistent_workers=bool(cfg.get("persistent_workers", True)),
        )
    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=workers,
        pin_memory=True,
        drop_last=len(train) >= batch_size,
        **kwargs,
    )
    val_loader = (
        DataLoader(
            val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            **kwargs,
        )
        if not world_size > 1 or rank == 0
        else None
    )
    test_loader = (
        DataLoader(
            test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            **kwargs,
        )
        if not world_size > 1 or rank == 0
        else None
    )
    summary = {
        "n_records": len(items),
        "n_train_records": len(train_idx),
        "n_val_records": len(val_idx),
        "n_test_records": len(test_idx),
        "n_train_windows": len(train),
        "n_val_windows": len(val),
        "n_test_windows": len(test),
        **split,
    }
    return train_loader, val_loader, test_loader, summary

class ChainStageB:
    def __init__(
        self,
        cfg: dict,
        device: str,
        encoder_path: str,
        *,
        attention: str,
        latent_grid: tuple[int, int],
    ):
        self.cfg = cfg
        self.section = cfg["predictor_training"]
        self.device = device
        self.context = int(cfg["context"])
        self.latent_grid = tuple(int(x) for x in latent_grid)
        self.image_size = tuple(int(x) for x in cfg["tactile"]["img_size"])
        self.patch_size = int(cfg["tactile"].get("patch_size", 16))
        self.source_grid = (
            self.image_size[0] // self.patch_size,
            self.image_size[1] // self.patch_size,
        )
        self.spatial_tokens = self.latent_grid[0] * self.latent_grid[1]
        self.embed_dim = int(cfg["tactile"].get("embed_dim", 384))
        self.target_future_mode = str(
            self.section.get(
                "target_future_mode", "absolute_full_clip"
            )
        ).lower()
        if self.target_future_mode not in (
            "absolute_full_clip",
            "reindexed_frames",
        ):
            raise ValueError(
                "target_future_mode must be 'absolute_full_clip' or "
                "'reindexed_frames'"
            )
        self.target_encoder_source = str(
            self.section.get("target_encoder_source", "ema")
        ).lower()
        if self.target_encoder_source not in ("ema", "online"):
            raise ValueError(
                "target_encoder_source must be 'ema' or 'online'"
            )
        self.use_visual_history = bool(
            self.section.get("use_visual_history", True)
        )
        self.use_state_history = bool(
            self.section.get("use_state_history", True)
        )
        self.selection_mode = str(
            self.section.get("selection_mode", "composite")
        ).lower()
        if self.selection_mode not in ("composite", "tactile"):
            raise ValueError(
                "selection_mode must be 'composite' or 'tactile'"
            )

        state = torch.load(encoder_path, map_location=device, weights_only=False)
        if "encoder" not in state:
            raise ValueError("Stage-A checkpoint does not contain an encoder")
        common = dict(
            attention_mode=attention,
            embed_dim=self.embed_dim,
            depth=int(cfg["tactile"].get("depth", 6)),
            heads=int(cfg["tactile"].get("num_heads", 6)),
            image_size=self.image_size,
            max_time=int(cfg["context"]) + int(cfg["horizon"]),
            causal_depth=int(
                cfg.get("encoder_training", {}).get(
                    "causal_temporal_depth", 2
                )
            ),
            drop_path=float(cfg["tactile"].get("drop_path", 0.1)),
        )
        self.encoder = ChainTactileEncoder(**common).to(device)
        self.encoder_target = ChainTactileEncoder(**common).to(device)
        self.encoder.load_state_dict(
            _align_temporal_embedding(
                state["encoder"], self.encoder.state_dict()
            )
        )
        self.encoder_target.load_state_dict(
            _align_temporal_embedding(
                (
                    state["encoder"]
                    if self.target_encoder_source == "online"
                    else state.get("encoder_target", state["encoder"])
                ),
                self.encoder_target.state_dict(),
            )
        )
        for module in (self.encoder, self.encoder_target):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        self.visual = _build_visual(cfg, self.section, device)
        self.visual_dim = int(
            getattr(
                self.visual,
                "embed_dim",
                getattr(getattr(self.visual, "backbone", None), "embed_dim", 768),
            )
        )
        self.state_dim = int(
            self.section.get("state_history_dim", cfg.get("state_history_dim", 8))
        )
        self.future_offsets = tuple(
            int(x)
            for x in cfg.get(
                "future_offsets", range(1, int(cfg["horizon"]) + 1)
            )
        )
        self.visual_future_offsets = tuple(
            int(x)
            for x in cfg.get(
                "visual_future_offsets", range(2, int(cfg["horizon"]) + 1, 2)
            )
        )
        self.predictor = ChainWorldPredictor(
            tactile_dim=self.embed_dim,
            visual_dim=self.visual_dim,
            state_dim=self.state_dim,
            spatial_tokens=self.spatial_tokens,
            hidden_dim=int(self.section.get("predictor_dim", 384)),
            depth=int(self.section.get("predictor_depth", 2)),
            condition_depth=int(self.section.get("condition_depth", 2)),
            heads=int(self.section.get("predictor_heads", 6)),
            max_time=int(cfg["context"]) + int(cfg["horizon"]) + 1,
            future_offsets=self.future_offsets,
            visual_future_offsets=self.visual_future_offsets,
            num_condition_tokens=int(
                self.section.get("num_condition_tokens", 4)
            ),
            projection_type=str(
                self.section.get("modality_projection", "linear")
            ),
            drop_path=float(self.section.get("drop_path", 0.05)),
        ).to(device)
        self.predictor_base = self.predictor
        self.contact_probe = nn.Sequential(
            nn.LayerNorm(2 * self.embed_dim),
            nn.Linear(2 * self.embed_dim, 1),
        ).to(device)
        self.slip_probe = nn.Sequential(
            nn.LayerNorm(2 * self.embed_dim),
            nn.Linear(2 * self.embed_dim, 1),
        ).to(device)
        self.opt = torch.optim.AdamW(
            list(self.predictor.parameters())
            + list(self.contact_probe.parameters())
            + list(self.slip_probe.parameters()),
            lr=float(self.section.get("lr", 5e-5)),
            weight_decay=float(self.section.get("weight_decay", 0.05)),
        )
        self.amp_dtype = (
            torch.bfloat16
            if str(self.section.get("amp_dtype", "bfloat16"))
            in ("bf16", "bfloat16")
            else torch.float16
        )
        self.use_amp = bool(self.section.get("amp", True)) and device.startswith(
            "cuda"
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp and self.amp_dtype == torch.float16,
        )
        self.state_scale = torch.as_tensor(
            cfg.get("state_scale", [1.0] * self.state_dim),
            device=device,
            dtype=torch.float32,
        ).view(1, 1, 1, -1)
        if self.state_scale.shape[-1] != self.state_dim:
            raise ValueError("state_scale width does not match state history")

    def _encode_visual(
        self, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video = batch["vision_arms"].to(self.device, non_blocking=True)
        batch_size, arms, time, channels, height, width = video.shape
        tubelet = int(self.cfg.get("visual_tubelet_size", 2))
        steps = time // tubelet
        context_steps = self.context // tubelet
        if not self.use_visual_history:
            zeros = torch.zeros(
                batch_size,
                arms,
                self.visual_dim,
                device=self.device,
                dtype=torch.float32,
            )
            return (
                zeros[:, :, None].expand(-1, -1, context_steps, -1),
                zeros[:, :, None].expand(
                    -1, -1, len(self.visual_future_offsets), -1
                ),
                zeros[:, :, None].expand(-1, -1, context_steps, -1),
            )
        flat = _prepare_visual_gpu(
            video.reshape(batch_size * arms, time, channels, height, width),
            int(self.cfg.get("crop_size", 384)),
        )
        flat = flat.permute(0, 2, 1, 3, 4).contiguous()
        with torch.no_grad():
            tokens = _visual_tokens(
                _run_visual(self.visual, flat),
                batch_size * arms,
                steps,
            )
        tokens = tokens.reshape(
            batch_size, arms, steps, tokens.shape[-2], tokens.shape[-1]
        )
        context = tokens[:, :, :context_steps].mean(dim=-2)
        future_indices = [
            context_steps + offset // tubelet - 1
            for offset in self.visual_future_offsets
        ]
        future = tokens[:, :, future_indices].mean(dim=-2)
        return context, future, tokens[:, :, :context_steps]

    def _prepare_tactile(
        self, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        chains = batch["tactile_chains"].to(self.device, non_blocking=True)
        if chains.dtype == torch.uint8:
            chains = chains.float().div_(255.0)
        else:
            chains = chains.float()
        batch_size, count, time, channels, height, width = chains.shape
        flat = chains.reshape(batch_size * count, time, channels, height, width)
        flat = _resize_clip(flat, self.image_size)
        encoded_channels = flat.shape[2] * 2
        flat = _causal_delta_clip(flat)
        return flat.reshape(
            batch_size, count, time, encoded_channels, *self.image_size
        ), chains

    def _state_history(self, batch: dict) -> torch.Tensor:
        state = batch["state_arms"].to(self.device, non_blocking=True).float()
        state = state[:, :, : self.context]
        if not self.use_state_history:
            return torch.zeros_like(state)
        return state / self.state_scale

    def _probe_losses(
        self,
        current: torch.Tensor,
        batch: dict,
        chain_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        pooled = torch.cat((current.mean(dim=-2), current.amax(dim=-2)), dim=-1)
        contact_logits = self.contact_probe(pooled).squeeze(-1)
        slip_logits = self.slip_probe(pooled).squeeze(-1)
        index = self.context - 1
        contact_target = batch["contact"].to(self.device)[:, index].float()
        slip_target = batch["slip"].to(self.device)[:, index].float()
        mask = chain_valid.to(contact_logits.dtype)

        def masked_bce(logits, target):
            error = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )
            return (error * mask).sum() / mask.sum().clamp_min(1.0)

        contact_loss = masked_bce(contact_logits, contact_target)
        slip_loss = masked_bce(slip_logits, slip_target)
        return (
            contact_loss,
            slip_loss,
            _auroc(contact_logits[chain_valid], contact_target[chain_valid]),
            _auroc(slip_logits[chain_valid], slip_target[chain_valid]),
        )

    def step(self, batch: dict, train: bool) -> dict[str, float]:
        clip, _ = self._prepare_tactile(batch)
        chain_valid = batch["chain_valid"].to(self.device, dtype=torch.bool)
        if train:
            chain_dropout = float(
                self.section.get("stage_b_chain_dropout", 0.0)
            )
            arm_dropout = float(
                self.section.get("stage_b_arm_dropout", 0.0)
            )
            if chain_dropout > 0 or arm_dropout > 0:
                keep = chain_valid.clone()
                if chain_dropout > 0:
                    keep &= torch.rand_like(chain_valid, dtype=torch.float32) >= chain_dropout
                if arm_dropout > 0:
                    drop_arm = torch.rand(
                        chain_valid.shape[0],
                        self.predictor_base.num_arms,
                        device=self.device,
                    ) < arm_dropout
                    drop_chain = drop_arm[:, [0, 0, 1, 1]]
                    keep &= ~drop_chain
                missing = ~keep.any(dim=1)
                if missing.any():
                    fallback = chain_valid.float().multinomial(1).squeeze(1)
                    keep[missing, fallback[missing]] = True
                chain_valid = keep
        active_arm = torch.stack(
            (
                chain_valid[:, 0] | chain_valid[:, 1],
                chain_valid[:, 2] | chain_valid[:, 3],
            ),
            dim=1,
        )
        manifest_arm_valid = batch["arm_valid"].to(
            self.device, dtype=torch.bool
        )
        arm_valid = active_arm & manifest_arm_valid
        state_history = self._state_history(batch)
        with torch.autocast(
            device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
        ):
            context_indices = list(range(self.context))
            future_indices = [
                self.context + offset - 1 for offset in self.future_offsets
            ]
            tactile_context = _encode_chains(
                self.encoder,
                clip,
                context_indices,
                self.source_grid,
                self.latent_grid,
            )
            with torch.no_grad():
                if self.target_future_mode == "absolute_full_clip":
                    target_all = _encode_full_chains(
                        self.encoder_target,
                        clip,
                        self.source_grid,
                        self.latent_grid,
                    )
                    target_context = target_all[:, :, : self.context]
                    target_future = target_all[:, :, future_indices]
                else:
                    target_context = _encode_chains(
                        self.encoder_target,
                        clip,
                        context_indices,
                        self.source_grid,
                        self.latent_grid,
                    )
                    target_future = _encode_chains(
                        self.encoder_target,
                        clip,
                        future_indices,
                        self.source_grid,
                        self.latent_grid,
                    )
                visual_context, visual_target, visual_context_tokens = (
                    self._encode_visual(batch)
                )
            output = self.predictor(
                tactile_context,
                context_indices,
                chain_valid,
                visual_context,
                list(range(visual_context.shape[2])),
                arm_valid,
                state_history,
                self.context,
            )
            future_prediction = output["future_tactile"].permute(0, 2, 1, 3, 4)
            target_future_time = target_future.permute(0, 2, 1, 3, 4)
            current_target = target_context[:, :, -1]
            tactile_delta_loss, tactile_delta_copy, delta_rms = (
                _dual_tactile_delta_loss(
                    future_prediction,
                    target_future_time,
                    current_target,
                    scale=float(self.section.get("delta_loss_scale", 0.01)),
                    direction_weight=float(
                        self.section.get("delta_direction_weight", 0.1)
                    ),
                    hand_valid=chain_valid,
                )
            )
            tactile_norm_error = (
                F.normalize(future_prediction, dim=-1)
                - F.normalize(target_future_time, dim=-1)
            ).square().mean(dim=(-1, -2))
            tactile_chain_mask = chain_valid[:, None, :].expand_as(
                tactile_norm_error
            ).to(tactile_norm_error.dtype)
            tactile_normalized_mse = (
                tactile_norm_error * tactile_chain_mask
            ).sum() / tactile_chain_mask.sum().clamp_min(1.0)
            tactile_cosine = (
                F.cosine_similarity(
                    future_prediction, target_future_time, dim=-1
                ).mean(dim=-1)
                * tactile_chain_mask
            ).sum() / tactile_chain_mask.sum().clamp_min(1.0)

            visual_reference = visual_context[:, :, -1]
            visual_error = (
                F.normalize(output["future_visual"], dim=-1)
                - F.normalize(visual_target, dim=-1)
            ).square().mean(dim=-1)
            visual_copy_error = (
                F.normalize(
                    visual_reference[:, :, None].expand_as(visual_target),
                    dim=-1,
                )
                - F.normalize(visual_target, dim=-1)
            ).square().mean(dim=-1)
            arm_mask = arm_valid[:, :, None].to(visual_error.dtype)
            visual_loss = (visual_error * arm_mask).sum() / arm_mask.sum().clamp_min(1.0)
            visual_copy = (visual_copy_error * arm_mask).sum() / arm_mask.sum().clamp_min(1.0)
            visual_target_delta = (
                visual_target - visual_reference[:, :, None]
            )
            visual_scale = visual_target_delta.square().mean().sqrt().clamp_min(1e-6)
            visual_delta = F.smooth_l1_loss(
                (output["future_visual"] - visual_reference[:, :, None])
                / visual_scale,
                visual_target_delta / visual_scale,
                reduction="none",
                beta=0.2,
            ).mean(dim=-1)
            visual_delta_copy_error = F.smooth_l1_loss(
                torch.zeros_like(visual_target_delta),
                visual_target_delta / visual_scale,
                reduction="none",
                beta=0.2,
            ).mean(dim=-1)
            visual_delta_loss = (
                visual_delta * arm_mask
            ).sum() / arm_mask.sum().clamp_min(1.0)
            visual_delta_copy = (
                visual_delta_copy_error * arm_mask
            ).sum() / arm_mask.sum().clamp_min(1.0)
            visual_total = visual_loss + float(
                self.section.get("visual_delta_weight", 0.5)
            ) * visual_delta_loss

            history_summary_target = _summary_4way(
                target_context.permute(0, 2, 1, 3, 4)
            )
            future_summary_target = _summary_4way(
                target_future_time
            )
            history_summary_loss = _normalized_summary_loss(
                output["condition_history_summary"],
                history_summary_target[:, None].expand_as(
                    output["condition_history_summary"]
                ),
                hand_valid=chain_valid,
                hand_dim=2,
            )
            future_summary_loss = _normalized_summary_loss(
                output["condition_future_summary"],
                future_summary_target[:, None].expand_as(
                    output["condition_future_summary"]
                ),
                hand_valid=chain_valid,
                hand_dim=2,
            )
            future_trajectory_target = target_future_time.mean(dim=-2)
            future_trajectory_loss = _normalized_summary_loss(
                output["condition_future_trajectory"],
                future_trajectory_target,
                hand_valid=chain_valid,
                hand_dim=2,
            )
            future_visual_target = _temporal_2way(
                visual_target.flatten(0, 1)
            ).reshape(
                visual_target.shape[0],
                visual_target.shape[1],
                2,
                visual_target.shape[-1],
            )
            future_visual_summary_loss = _normalized_summary_loss(
                output["condition_future_visual_summary"],
                future_visual_target[:, None].expand_as(
                    output["condition_future_visual_summary"]
                ),
                hand_valid=arm_valid,
                hand_dim=2,
            )
            history_state_target = _temporal_2way(
                state_history.flatten(0, 1)
            ).reshape(
                state_history.shape[0],
                state_history.shape[1],
                2,
                state_history.shape[-1],
            )
            state_summary_loss = _normalized_summary_loss(
                output["condition_history_state_summary"],
                history_state_target[:, None].expand_as(
                    output["condition_history_state_summary"]
                ),
                hand_valid=arm_valid,
                hand_dim=2,
            )
            condition_summary_loss = (
                history_summary_loss
                + future_summary_loss
                + future_trajectory_loss
                + future_visual_summary_loss
                + state_summary_loss
            )
            condition_diversity = _token_diversity_loss(
                output["condition_tokens"]
            )
            contact_loss, slip_loss, contact_auc, slip_auc = self._probe_losses(
                tactile_context[:, :, -1], batch, chain_valid
            )
            loss = (
                tactile_delta_loss
                + float(self.section.get("lambda_visual", 0.5))
                * visual_total
                + float(self.section.get("lambda_condition", 0.5))
                * condition_summary_loss
                + float(self.section.get("lambda_condition_diversity", 0.02))
                * condition_diversity
                + float(self.section.get("lambda_contact", 0.05))
                * contact_loss
                + float(self.section.get("lambda_slip", 0.05)) * slip_loss
            )

        relative = (
            float(tactile_delta_loss.detach()) + 1e-6
        ) / (float(tactile_delta_copy.detach()) + 1e-6)
        visual_relative = (
            float(visual_total.detach()) + 1e-6
        ) / (
            float(visual_copy + float(self.section.get("visual_delta_weight", 0.5))
            * visual_delta_copy)
            + 1e-6
        )
        query_std = (
            output["condition_tokens"].detach().std(dim=1).mean()
            if self.predictor_base.num_condition_tokens > 1
            else torch.zeros((), device=output["condition_tokens"].device)
        )
        single_arm = (arm_valid.sum(dim=1) == 1).float().mean()
        dual_arm = (arm_valid.sum(dim=1) == 2).float().mean()
        selection = relative
        if self.selection_mode == "composite":
            selection += (
                float(self.section.get("selection_visual_weight", 0.2))
                * visual_relative
                + float(self.section.get("selection_condition_weight", 0.1))
                * float(condition_summary_loss.detach())
            )
        selection += (
            float(self.section.get("probe_selection_weight", 0.001))
            * (2 - contact_auc - slip_auc)
        )
        metrics = {
            "loss": float(loss.detach()),
            "tactile_relative": relative,
            "tactile_delta": float(tactile_delta_loss.detach()),
            "tactile_copy": float(tactile_delta_copy.detach()),
            "delta_rms": float(delta_rms),
            "tactile_normalized_mse": float(tactile_normalized_mse.detach()),
            "tactile_cosine": float(tactile_cosine.detach()),
            "visual_relative": visual_relative,
            "visual_delta_relative": (
                float(visual_delta_loss.detach()) + 1e-6
            ) / (float(visual_delta_copy.detach()) + 1e-6),
            "condition_loss": float(condition_summary_loss.detach()),
            "condition_history": float(history_summary_loss.detach()),
            "condition_future": float(future_summary_loss.detach()),
            "condition_trajectory": float(future_trajectory_loss.detach()),
            "condition_visual": float(future_visual_summary_loss.detach()),
            "condition_state": float(state_summary_loss.detach()),
            "condition_diversity": float(condition_diversity.detach()),
            "query_token_std": float(query_std),
            "contact_probe_auroc": contact_auc,
            "slip_probe_auroc": slip_auc,
            "active_chain_rate": float(chain_valid.float().mean()),
            "active_arm_rate": float(arm_valid.float().mean()),
            "single_arm_rate": float(single_arm),
            "dual_arm_rate": float(dual_arm),
            "selection_loss": selection,
            "selection_mode_tactile": float(
                self.selection_mode == "tactile"
            ),
        }
        if not train:
            self._eval_features = {
                "condition_tokens": output["condition_tokens"].detach().cpu(),
                "history_latent": target_context.mean(dim=(2, 3)).detach().cpu(),
                "future_latent": target_future.mean(dim=(2, 3)).detach().cpu(),
                "predicted_future_latent": (
                    future_prediction.mean(dim=(1, 3)).detach().cpu()
                ),
                "chain_valid": chain_valid.detach().cpu(),
                "arm_valid": arm_valid.detach().cpu(),
                "history_visual": visual_context.mean(dim=2).detach().cpu(),
                "future_visual": visual_target.mean(dim=2).detach().cpu(),
            }
        if train:
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(
                list(self.predictor.parameters())
                + list(self.contact_probe.parameters())
                + list(self.slip_probe.parameters()),
                1.0,
            )
            self.scaler.step(self.opt)
            self.scaler.update()
        return metrics

    def state_dict(self) -> dict:
        return {
            "predictor": self.predictor_base.state_dict(),
            "contact_probe": self.contact_probe.state_dict(),
            "slip_probe": self.slip_probe.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
        }

def evaluate(trainer: ChainStageB, loader, max_batches: int) -> dict[str, float]:
    if loader is None:
        return {}
    trainer.predictor.eval()
    trainer.contact_probe.eval()
    trainer.slip_probe.eval()
    records = []
    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            records.append(trainer.step(batch, train=False))
            if max_batches and index >= max_batches:
                break
    trainer.predictor.train()
    trainer.contact_probe.train()
    trainer.slip_probe.train()
    return _average(records)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None, help="cuda, cuda:N, or cpu")
    parser.add_argument("--attention", required=True, choices=("per_frame_causal", "full_clip_bidirectional"))
    parser.add_argument("--latent-grid", required=True, help="HxW")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--predictor-depth", type=int, default=None)
    parser.add_argument("--condition-depth", type=int, default=None)
    parser.add_argument("--condition-tokens", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--visual-weight", type=float, default=None)
    parser.add_argument("--condition-weight", type=float, default=None)
    parser.add_argument("--query-diversity", type=float, default=None)
    parser.add_argument(
        "--projection",
        choices=("linear", "mlp", "normed_mlp"),
        default=None,
    )
    parser.add_argument("--chain-dropout", type=float, default=None)
    parser.add_argument("--arm-dropout", type=float, default=None)
    parser.add_argument("--delta-scale", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--sharing-strategy",
        choices=("file_system", "file_descriptor"),
        default=None,
    )
    parser.add_argument("--early-min-steps", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--reset-best", action="store_true")
    parser.add_argument("--test-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument(
        "--target-future-mode",
        choices=("absolute_full_clip", "reindexed_frames"),
        default=None,
    )
    parser.add_argument(
        "--target-encoder-source",
        choices=("ema", "online"),
        default=None,
    )
    parser.add_argument("--tactile-only", action="store_true")
    parser.add_argument(
        "--selection-mode",
        choices=("composite", "tactile"),
        default=None,
    )
    args = parser.parse_args()
    cfg = _load_config(args.config)
    if args.batch_size:
        cfg["predictor_training"]["batch_size"] = args.batch_size
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.predictor_depth:
        cfg["predictor_training"]["predictor_depth"] = args.predictor_depth
    if args.condition_depth:
        cfg["predictor_training"]["condition_depth"] = args.condition_depth
    if args.condition_tokens is not None:
        if args.condition_tokens <= 0:
            parser.error("--condition-tokens must be positive")
        cfg["predictor_training"]["num_condition_tokens"] = int(
            args.condition_tokens
        )
    if args.lr:
        cfg["predictor_training"]["lr"] = args.lr
    if args.visual_weight is not None:
        cfg["predictor_training"]["lambda_visual"] = args.visual_weight
    if args.condition_weight is not None:
        cfg["predictor_training"]["lambda_condition"] = args.condition_weight
    if args.query_diversity is not None:
        cfg["predictor_training"]["lambda_condition_diversity"] = (
            args.query_diversity
        )
    if args.projection:
        cfg["predictor_training"]["modality_projection"] = args.projection
    if args.chain_dropout is not None:
        cfg["predictor_training"]["stage_b_chain_dropout"] = (
            args.chain_dropout
        )
    if args.arm_dropout is not None:
        cfg["predictor_training"]["stage_b_arm_dropout"] = args.arm_dropout
    if args.delta_scale is not None:
        cfg["predictor_training"]["delta_loss_scale"] = args.delta_scale
    if args.num_workers is not None:
        cfg["num_workers"] = int(args.num_workers)
    if args.sharing_strategy is not None:
        cfg["tensor_sharing_strategy"] = args.sharing_strategy
    if args.early_min_steps is not None:
        cfg["predictor_training"]["early_stopping_min_steps"] = int(
            args.early_min_steps
        )
    if args.patience is not None:
        cfg["predictor_training"]["early_stopping_patience"] = int(
            args.patience
        )
    if args.max_test_batches is not None:
        cfg["predictor_training"]["max_test_batches"] = int(
            args.max_test_batches
        )
    if args.max_val_batches is not None:
        cfg["predictor_training"]["max_val_batches"] = int(
            args.max_val_batches
        )
    if args.target_future_mode is not None:
        cfg["predictor_training"]["target_future_mode"] = (
            args.target_future_mode
        )
    if args.target_encoder_source is not None:
        cfg["predictor_training"]["target_encoder_source"] = (
            args.target_encoder_source
        )
    if args.tactile_only:
        cfg["predictor_training"]["use_visual_history"] = False
        cfg["predictor_training"]["use_state_history"] = False
    if args.selection_mode is not None:
        cfg["predictor_training"]["selection_mode"] = args.selection_mode
    height, width = args.latent_grid.lower().split("x", 1)
    grid = (int(height), int(width))
    rank, world_size, device = setup_distributed(args.device)
    seed = int(cfg.get("seed", 0))
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)
    torch.multiprocessing.set_sharing_strategy(
        str(cfg.get("tensor_sharing_strategy", "file_descriptor"))
    )
    requested = Path(args.output)
    output = (
        Path(
            ensure_writable_output(
                unique_output_dir(requested, resume=args.resume),
                cfg.get("protected_data_roots", ()),
            )
        )
        if rank == 0
        else requested
    )
    if world_size > 1:
        values = [str(output) if rank == 0 else None]
        dist.broadcast_object_list(values, src=0)
        output = Path(values[0])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    trainer = ChainStageB(
        cfg,
        device,
        args.encoder_checkpoint,
        attention=args.attention,
        latent_grid=grid,
    )
    module = trainer.predictor
    if world_size > 1:
        module = DistributedDataParallel(
            module,
            device_ids=[torch.cuda.current_device()],
            find_unused_parameters=False,
        )
        trainer.predictor = module
    section = cfg["predictor_training"]
    batch_size = int(section.get("batch_size", 1))
    train_loader, val_loader, test_loader, split = _build_loaders(
        cfg, batch_size=batch_size, rank=rank, world_size=world_size
    )
    steps = int(args.max_steps or section.get("max_steps", 10000))
    start_step = 0
    best = math.inf
    best_step = 0
    if args.resume:
        resume_state = torch.load(
            args.resume, map_location=device, weights_only=False
        )
        trainer.predictor_base.load_state_dict(resume_state["predictor"])
        trainer.contact_probe.load_state_dict(
            resume_state["contact_probe"]
        )
        trainer.slip_probe.load_state_dict(resume_state["slip_probe"])
        if resume_state.get("optimizer") is not None:
            trainer.opt.load_state_dict(resume_state["optimizer"])
        if (
            resume_state.get("scaler") is not None
            and trainer.scaler.is_enabled()
        ):
            trainer.scaler.load_state_dict(resume_state["scaler"])
        start_step = int(resume_state.get("step", 0))
        best = float(resume_state.get("score", best))
        best_step = start_step
        if args.reset_best:
            best = math.inf
        if not args.eval_only and start_step >= steps:
            raise ValueError(
                f"resume step {start_step} is not below max_steps={steps}"
            )
    scheduler = WarmupCosineMultiplier(
        trainer.opt,
        steps,
        int(section.get("warmup_steps", 200)),
        min_lr_scale=float(section.get("min_lr_scale", 0.02)),
    )
    scheduler.jump_to(start_step)
    if args.eval_only:
        test_batches = int(
            args.test_batches
            if args.test_batches is not None
            else section.get("max_test_batches", 0)
        )
        test = evaluate(trainer, test_loader, test_batches)
        if rank == 0:
            (output / "test_metrics.json").write_text(
                json.dumps(
                    {"test_metrics": test, "split": split},
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            print(f"[stageB] eval-only finished: {test}", flush=True)
        if world_size > 1:
            dist.barrier()
        return
    bad = 0
    records = {}
    min_steps = int(section.get("early_stopping_min_steps", 3000))
    patience = int(section.get("early_stopping_patience", 8))
    check_every = int(section.get("checkpoint_every_steps", 250))
    step = start_step
    if rank == 0:
        (output / "split.json").write_text(
            json.dumps(split, indent=2), encoding="utf-8"
        )
        resolved = dict(cfg)
        resolved["stage_a_encoder_checkpoint"] = args.encoder_checkpoint
        resolved["stage_a_attention"] = args.attention
        resolved["stage_a_latent_grid"] = list(grid)
        (output / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )
    while step < steps:
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(step)
        for batch in train_loader:
            step += 1
            metrics = trainer.step(batch, train=True)
            scheduler.step()
            if rank == 0 and (
                step == 1 or step % int(section.get("log_every", 20)) == 0
            ):
                print(
                    f"[stageB] step={step} "
                    + " ".join(
                        f"{key}={value:.5f}" for key, value in metrics.items()
                    ),
                    flush=True,
                )
            if step % check_every == 0 or step >= steps:
                stop = False
                if rank == 0:
                    val = evaluate(
                        trainer,
                        val_loader,
                        int(section.get("max_val_batches", 32)),
                    )
                    score = float(val["selection_loss"])
                    payload = {
                        "step": step,
                        "score": score,
                        "validation": val,
                        **trainer.state_dict(),
                    }
                    torch.save(payload, output / "latest_predictor.pt")
                    if score < best:
                        best = score
                        best_step = step
                        bad = 0
                        torch.save(payload, output / "best_predictor.pt")
                    elif step >= min_steps:
                        bad += 1
                    stop = step >= min_steps and bad >= patience
                    records[str(step)] = {
                        "score": score,
                        "best_step": best_step,
                        "bad_checks": bad,
                        "validation": val,
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
                    print(
                        f"[stageB] val step={step} score={score:.6f} "
                        f"tactile_rel={val['tactile_relative']:.6f} "
                        f"visual_rel={val['visual_relative']:.6f} "
                        f"query_loss={val['condition_loss']:.6f} "
                        f"best={best_step}@{best:.6f} bad={bad}/{patience}",
                        flush=True,
                    )
                if world_size > 1:
                    stop_tensor = torch.tensor(
                        [int(stop)], device=device, dtype=torch.int32
                    )
                    dist.broadcast(stop_tensor, src=0)
                    stop = bool(stop_tensor.item())
                if stop:
                    break
            if step >= steps:
                break
        else:
            continue
        break
    if rank == 0 and test_loader is not None:
        state = torch.load(
            output / "best_predictor.pt",
            map_location=device,
            weights_only=False,
        )
        trainer.predictor_base.load_state_dict(state["predictor"])
        trainer.contact_probe.load_state_dict(state["contact_probe"])
        trainer.slip_probe.load_state_dict(state["slip_probe"])
        test = evaluate(
            trainer,
            test_loader,
            int(section.get("max_test_batches", section.get("max_val_batches", 32))),
        )
        (output / "test_metrics.json").write_text(
            json.dumps(
                {"test_metrics": test, "split": split},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"[stageB] finished best={best_step}@{best}", flush=True)
    if world_size > 1:
        dist.barrier()


if __name__ == "__main__":
    main()
