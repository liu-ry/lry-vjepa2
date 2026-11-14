"""Two-stage tactile encoder / world-predictor training.

Stage A trains a shared-weight tactile encoder independently on left and right
sensor streams.  Stage B freezes that encoder and learns a dual-hand predictor
with a short-horizon event head and a sparse planning head.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    Sampler,
    SubsetRandomSampler,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.vjepa_2_1.data import RobotVisionTactileDataset
from app.vjepa_2_1.models.dense_tactile_predictor import DenseTactilePredictor
from app.vjepa_2_1.models.tactile_encoder import TactileEncoder
from app.vjepa_2_1.models.tactile_ssl import TactilePatchDecoder
from app.vjepa_2_1.models.touch_latent_forecaster import MaskedTouchLatentPredictor
from app.vjepa_2_1.tactile_training_utils import (
    WarmupCosineMultiplier,
    _covariance_loss,
    _encode_tactile_indices,
    _load_checkpoint,
    _run_visual,
    _variance_loss,
    _visual_tokens,
    distributed_barrier,
    ensure_writable_output,
    is_main_process,
    setup_distributed,
    unique_output_dir,
    update_ema,
)


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    for key in ("manifest", "context", "horizon", "tactile"):
        if key not in cfg:
            raise ValueError(f"config is missing {key!r}")
    return cfg


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            if not isinstance(node[part], dict):
                raise ValueError(f"override path {key!r} crosses a scalar")
            node = node[part]
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def _image_size(cfg: dict) -> tuple[int, int]:
    size = cfg["tactile"].get("img_size")
    if size is None:
        raise ValueError("tactile.img_size is required for the two-stage pipeline")
    if isinstance(size, int):
        return size, size
    if len(size) != 2:
        raise ValueError("tactile.img_size must be int or [height, width]")
    return int(size[0]), int(size[1])


def _build_encoder(cfg: dict, device: str) -> TactileEncoder:
    t = cfg["tactile"]
    channels = int(t.get("in_channels", 3))
    if bool(t.get("use_delta", True)):
        channels *= 2
    return TactileEncoder(
        embed_dim=int(t.get("embed_dim", 384)),
        in_channels=channels,
        patch_size=int(t.get("patch_size", 14)),
        tubelet_size=1,
        img_size=_image_size(cfg),
        num_frames=int(cfg["context"]) + int(cfg["horizon"]),
        depth=int(t.get("depth", 6)),
        num_heads=t.get("num_heads"),
        input_dim=t.get("input_dim"),
        taxel_grid=t.get("taxel_grid"),
        use_rope=bool(t.get("use_rope", True)),
        drop_path=float(t.get("drop_path", 0.0)),
        backbone=t.get("backbone"),
    ).to(device)


def _build_visual(cfg: dict, section: dict, device: str):
    from app.vjepa_2_1.utils import init_video_model

    model_cfg = cfg.get("model", {})
    context, horizon = int(cfg["context"]), int(cfg["horizon"])
    visual, _ = init_video_model(
        device=device,
        model_name=model_cfg.get("model_name", "vit_base"),
        patch_size=int(model_cfg.get("patch_size", 16)),
        max_num_frames=context + horizon,
        tubelet_size=int(cfg.get("visual_tubelet_size", 2)),
        crop_size=int(cfg.get("crop_size", 384)),
        use_rope=bool(model_cfg.get("use_rope", True)),
        use_sdpa=bool(model_cfg.get("use_sdpa", True)),
        modality_embedding=True,
        img_temporal_dim_size=int(model_cfg.get("img_temporal_dim_size", 1)),
        interpolate_rope=bool(model_cfg.get("interpolate_rope", True)),
        build_predictor=False,
    )
    _load_checkpoint(visual, section["vjepa_checkpoint"])
    visual.eval()
    for parameter in visual.parameters():
        parameter.requires_grad_(False)
    return visual


def _valid_tactile_items(items: list[dict]) -> list[dict]:
    selected = []
    for item in items:
        tactile = item.get("tactile")
        paths = (
            [tactile]
            if isinstance(tactile, (str, Path))
            else list(tactile) if isinstance(tactile, list) else []
        )
        if 1 <= len(paths) <= 2:
            selected.append(item)
    if not selected:
        raise ValueError("no manifest records contain one or two tactile streams")
    return selected


def _three_way_group_split(
    items: list[dict],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    stable: bool = False,
) -> tuple[list[int], list[int], list[int], dict]:
    groups = {}
    for index, item in enumerate(items):
        groups.setdefault(str(item.get("group", item.get("vision"))), []).append(index)
    names = sorted(groups)
    if len(names) < 3:
        raise ValueError("three-way split requires at least three groups")
    if stable:
        val_limit = float(val_ratio)
        test_limit = val_limit + float(test_ratio)
        if not 0.0 <= val_limit or test_limit >= 1.0:
            raise ValueError("stable split ratios must sum to less than one")
        test_groups, val_groups, train_groups = set(), set(), set()
        for name in names:
            digest = hashlib.sha256(
                f"{int(seed)}:{name}".encode("utf-8")
            ).digest()
            position = int.from_bytes(digest[:8], "big") / 2**64
            if position < val_limit:
                val_groups.add(name)
            elif position < test_limit:
                test_groups.add(name)
            else:
                train_groups.add(name)
        if not test_groups or not val_groups or not train_groups:
            raise ValueError(
                "stable split produced an empty partition; increase dataset size"
            )
    else:
        rng = random.Random(int(seed))
        rng.shuffle(names)
        total = len(names)
        n_test = min(max(1, int(round(total * float(test_ratio)))), total - 2)
        n_val = min(
            max(1, int(round(total * float(val_ratio)))),
            total - n_test - 1,
        )
        test_groups = set(names[:n_test])
        val_groups = set(names[n_test:n_test + n_val])
        train_groups = set(names[n_test + n_val:])
    split = {"train": [], "val": [], "test": []}
    for name, indices in groups.items():
        if name in test_groups:
            split["test"].extend(indices)
        elif name in val_groups:
            split["val"].extend(indices)
        else:
            split["train"].extend(indices)
    for key in split:
        split[key].sort()
    info = {
        "val_groups": sorted(val_groups),
        "test_groups": sorted(test_groups),
    }
    return split["train"], split["val"], split["test"], info


class TaskBalancedDistributedSampler(Sampler[int]):
    """Task-aware shuffling without changing the underlying dataset.

    ``power=0`` is uniform sampling, ``power=0.5`` is a conservative square-root
    inverse-frequency correction, and ``power=1`` fully balances task mass.
    Each rank draws a disjoint partition from the same seeded permutation.
    """

    def __init__(
        self,
        dataset,
        tasks: Sequence[str],
        *,
        num_replicas: int,
        rank: int,
        seed: int,
        power: float,
        drop_last: bool = True,
    ):
        if len(tasks) == 0:
            raise ValueError("task-balanced sampling requires task labels")
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("invalid sampler rank/world size")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.drop_last = bool(drop_last)
        self.total_size = len(dataset)
        if self.drop_last:
            self.num_samples = self.total_size // self.num_replicas
            self.total_draws = self.num_samples * self.num_replicas
        else:
            self.num_samples = math.ceil(self.total_size / self.num_replicas)
            self.total_draws = self.num_samples * self.num_replicas
        if self.num_samples == 0:
            raise ValueError("dataset is smaller than the distributed world size")
        counts = {}
        for task in tasks:
            counts[str(task)] = counts.get(str(task), 0) + 1
        windows = max(1, int(getattr(dataset, "windows_per_episode", 1)))
        per_window = [
            (1.0 / counts[str(task)]) ** float(power) for task in tasks
        ]
        self.weights = torch.tensor(
            [weight for weight in per_window for _ in range(windows)],
            dtype=torch.double,
        )
        if len(self.weights) != self.total_size:
            raise ValueError("task labels do not match dataset length")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_draws,
            replacement=True,
            generator=generator,
        )
        indices = indices.view(self.num_replicas, self.num_samples)[self.rank]
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


def _build_loaders(
    cfg: dict,
    *,
    stage: str,
    batch_size: int,
    distributed: bool,
    rank: int,
    world_size: int,
    max_records: int | None = None,
):
    manifest = cfg["manifest"]
    context, horizon = int(cfg["context"]), int(cfg["horizon"])
    load_state = stage == "predictor" and cfg.get("action_dim") is not None
    load_vision_frames = (
        context
        if stage == "predictor" and not bool(cfg.get("predict_visual", True))
        else None
    )
    require_contact = bool(cfg.get("require_contact", False))
    min_contact_frames = int(
        cfg.get(
            f"{stage}_min_contact_frames",
            cfg.get("min_contact_frames", 2),
        )
    )
    min_hand_contact_frames = int(
        cfg.get("min_hand_contact_frames", 1)
    )
    full = RobotVisionTactileDataset(
        manifest,
        context,
        horizon,
        load_horizon=horizon,
        load_vision=(stage == "predictor"),
        load_vision_frames=load_vision_frames,
        load_state=load_state,
        defer_image_conversion=True,
        vision_size=int(cfg.get("crop_size", 384)),
        normalize_vision=bool(cfg.get("normalize_vision", True)),
        random_crop=True,
        require_contact=require_contact,
        min_contact_frames=min_contact_frames,
        num_hands=2,
        min_hand_contact_frames=min_hand_contact_frames,
    )
    items = _valid_tactile_items(full.items)
    if require_contact:
        items = [
            item for item in items
            if len(full.contact_window_starts(item)) > 0
        ]
    if max_records is not None:
        items = items[:int(max_records)]
    train_idx, val_idx, test_idx, group_info = _three_way_group_split(
        items,
        val_ratio=float(cfg.get("val_ratio", 0.15)),
        test_ratio=float(cfg.get("test_ratio", 0.1)),
        seed=int(cfg.get("seed", 0)),
        stable=bool(cfg.get("stable_group_split", False)),
    )
    train_items = [items[index] for index in train_idx]
    val_items = [items[index] for index in val_idx]
    test_items = [items[index] for index in test_idx]
    windows = int(cfg.get("windows_per_episode", 4))
    val_windows = int(cfg.get("val_windows_per_episode", 2))
    train_ds = RobotVisionTactileDataset(
        manifest,
        context,
        horizon,
        items=train_items,
        load_horizon=horizon,
        load_vision=(stage == "predictor"),
        load_vision_frames=load_vision_frames,
        load_state=load_state,
        defer_image_conversion=True,
        vision_size=int(cfg.get("crop_size", 384)),
        normalize_vision=bool(cfg.get("normalize_vision", True)),
        random_crop=True,
        windows_per_episode=windows,
        require_contact=require_contact,
        min_contact_frames=min_contact_frames,
        num_hands=2,
        min_hand_contact_frames=min_hand_contact_frames,
    )
    sampler = None
    task_power = float(cfg.get("task_balance_power", 0.0))
    if task_power > 0:
        tasks = [
            str(item.get("task", item.get("group", item["vision"])))
            for item in train_items
        ]
        sampler = TaskBalancedDistributedSampler(
            train_ds,
            tasks,
            num_replicas=world_size,
            rank=rank,
            seed=int(cfg.get("seed", 0)),
            power=task_power,
            drop_last=True,
        )
    elif distributed:
        sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank,
            shuffle=True, seed=int(cfg.get("seed", 0)), drop_last=True,
        )
    loader_kwargs = {}
    workers = int(cfg.get("num_workers", 2))
    if workers > 0:
        loader_kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
        loader_kwargs["persistent_workers"] = bool(cfg.get("persistent_workers", True))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=len(train_ds) >= batch_size,
        **loader_kwargs,
    )
    val_loader = None
    if val_items and (not distributed or rank == 0):
        val_ds = RobotVisionTactileDataset(
            manifest,
            context,
            horizon,
            items=val_items,
            load_horizon=horizon,
            load_vision=(stage == "predictor"),
            load_vision_frames=load_vision_frames,
            load_state=load_state,
            defer_image_conversion=True,
            vision_size=int(cfg.get("crop_size", 384)),
            normalize_vision=bool(cfg.get("normalize_vision", True)),
            random_crop=False,
            windows_per_episode=val_windows,
            require_contact=require_contact,
            min_contact_frames=min_contact_frames,
        num_hands=2,
        min_hand_contact_frames=min_hand_contact_frames,
        )
        generator = torch.Generator().manual_seed(int(cfg.get("seed", 0)))
        order = torch.randperm(len(val_ds), generator=generator).tolist()
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(order),
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
            **loader_kwargs,
        )
    test_loader = None
    if test_items and (not distributed or rank == 0):
        test_ds = RobotVisionTactileDataset(
            manifest,
            context,
            horizon,
            items=test_items,
            load_horizon=horizon,
            load_vision=(stage == "predictor"),
            load_vision_frames=load_vision_frames,
            load_state=load_state,
            defer_image_conversion=True,
            vision_size=int(cfg.get("crop_size", 384)),
            normalize_vision=bool(cfg.get("normalize_vision", True)),
            random_crop=False,
            windows_per_episode=val_windows,
            require_contact=require_contact,
            min_contact_frames=min_contact_frames,
        num_hands=2,
        min_hand_contact_frames=min_hand_contact_frames,
        )
        generator = torch.Generator().manual_seed(int(cfg.get("seed", 0)) + 1)
        order = torch.randperm(len(test_ds), generator=generator).tolist()
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(order),
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
            **loader_kwargs,
        )
    split = {
        "n_records": len(items),
        "n_train_records": len(train_items),
        "n_val_records": len(val_items),
        "n_test_records": len(test_items),
        "n_train_windows": len(train_ds),
        "n_val_windows": 0 if val_loader is None else len(val_loader.dataset),
        "n_test_windows": 0 if test_loader is None else len(test_loader.dataset),
        **group_info,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
    }
    return train_loader, val_loader, test_loader, split


def _resize_frames(frames: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    batch, time, channels, height, width = frames.shape
    if (height, width) == size:
        return frames
    flat = frames.reshape(batch * time, channels, height, width)
    flat = F.interpolate(flat, size=size, mode="bilinear", align_corners=False)
    return flat.view(batch, time, channels, *size)


def _image_tensor_to_float(images: torch.Tensor) -> torch.Tensor:
    """Convert decoded uint8 frames on the accelerator."""
    if images.dtype == torch.uint8:
        return images.float().div_(255.0)
    images = images.float()
    if images.numel() and float(images.max()) > 1.5:
        images.div_(255.0)
    return images


def _prepare_visual_gpu(
    video: torch.Tensor,
    size: int,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    video = _image_tensor_to_float(video)
    if video.shape[-2:] != (size, size):
        batch, time, channels, height, width = video.shape
        flat = video.reshape(batch * time, channels, height, width)
        flat = F.interpolate(
            flat, size=(size, size), mode="bilinear", align_corners=False
        )
        video = flat.view(batch, time, channels, size, size)
    mean_t = torch.tensor(
        mean, device=video.device, dtype=video.dtype
    ).view(1, 1, -1, 1, 1)
    std_t = torch.tensor(
        std, device=video.device, dtype=video.dtype
    ).view(1, 1, -1, 1, 1)
    return (video - mean_t) / std_t


def _pool_spatial_grid(
    tokens: torch.Tensor,
    source_grid: tuple[int, int],
    target_grid: tuple[int, int],
) -> torch.Tensor:
    """Keep a coarse spatial map of latent tokens without full token blast."""
    if tokens.ndim not in (4, 5):
        raise ValueError("spatial tokens must be [B,T,N,D] or [B,T,H,N,D]")
    prefix = tokens.shape[:-2]
    source_h, source_w = (int(source_grid[0]), int(source_grid[1]))
    target_h, target_w = (int(target_grid[0]), int(target_grid[1]))
    if source_h * source_w != tokens.shape[-2]:
        raise ValueError(
            f"token grid {source_h}x{source_w} does not match "
            f"{tokens.shape[-2]} tokens"
        )
    dim = tokens.shape[-1]
    grid = tokens.reshape(*prefix, source_h, source_w, dim)
    grid = grid.permute(*range(len(prefix)), len(prefix) + 2,
                        len(prefix), len(prefix) + 1)
    flat = grid.reshape(-1, dim, source_h, source_w)
    flat = F.adaptive_avg_pool2d(flat, (target_h, target_w))
    flat = flat.reshape(*prefix, dim, target_h, target_w)
    flat = flat.permute(*range(len(prefix)), len(prefix) + 1,
                        len(prefix) + 2, len(prefix))
    return flat.reshape(*prefix, target_h * target_w, dim)


def _causal_tactile_pair(
    frames: torch.Tensor,
    indices: list[int],
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``[current, current-previous]`` and the resize-preserving target."""
    batch, _, hand_count, channels, height, width = frames.shape
    idx = torch.as_tensor(indices, device=frames.device, dtype=torch.long)
    if idx.max() >= frames.shape[1]:
        raise ValueError("tactile frame offset exceeds the loaded window")
    prev = (idx - 1).clamp_min(0)
    current = frames.index_select(1, idx).reshape(
        batch * len(indices) * hand_count, channels, height, width
    )
    previous = frames.index_select(1, prev).reshape_as(current)
    current = F.interpolate(
        current, size=image_size, mode="bilinear", align_corners=False
    ) if (height, width) != image_size else current
    previous = F.interpolate(
        previous, size=image_size, mode="bilinear", align_corners=False
    ) if (height, width) != image_size else previous
    current = current.view(batch, len(indices), hand_count, channels, *image_size)
    previous = previous.view(batch, len(indices), hand_count, channels, *image_size)
    delta = current - previous
    delta = delta * (idx > 0).to(delta.dtype).view(1, -1, 1, 1, 1, 1)
    pair = torch.cat((current, delta), dim=3)
    return pair, pair


def _encode_causal_indices(
    encoder: TactileEncoder,
    left: torch.Tensor,
    right: torch.Tensor,
    indices: list[int],
    image_size: tuple[int, int],
    *,
    use_delta: bool,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    hands = torch.stack((left, right), dim=2)
    pair, _ = _causal_tactile_pair(hands, indices, image_size)
    if not use_delta:
        pair = pair[:, :, :, : pair.shape[3] // 2]
    batch, time, hand_count, channels, height, width = pair.shape
    flat = pair.reshape(batch * time * hand_count, channels, height, width)
    flat_mask = None
    if token_mask is not None:
        expected = (batch, time, hand_count, encoder.max_spatial_tokens)
        if token_mask.shape != expected:
            raise ValueError(
                f"token_mask must have shape {expected}, got {tuple(token_mask.shape)}"
            )
        flat_mask = token_mask.reshape(
            batch * time * hand_count, token_mask.shape[-1]
        )
    tokens = encoder(flat, token_mask=flat_mask)
    return tokens.squeeze(1).reshape(
        batch, time, hand_count, encoder.max_spatial_tokens, encoder.embed_dim
    )


def _pool_spatial_grid_masked(
    tokens: torch.Tensor,
    valid: torch.Tensor,
    source_grid: tuple[int, int],
    target_grid: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool spatial tokens while excluding masked values.

    Args:
        tokens: ``[B,T,H,sh*sw,D]``.
        valid: ``[B,T,H,sh*sw]`` boolean, true where a token is visible.
    """
    if tokens.ndim != 5 or valid.shape != tokens.shape[:-1]:
        raise ValueError("masked spatial pooling received incompatible shapes")
    batch, time, hands, count, dim = tokens.shape
    source_h, source_w = (int(source_grid[0]), int(source_grid[1]))
    target_h, target_w = (int(target_grid[0]), int(target_grid[1]))
    if source_h * source_w != count:
        raise ValueError("source grid does not match tactile token count")
    flat_tokens = tokens.reshape(
        batch * time * hands, count, dim
    ).transpose(1, 2).reshape(batch * time * hands, dim, source_h, source_w)
    flat_valid = valid.reshape(batch * time * hands, count).reshape(
        batch * time * hands, 1, source_h, source_w
    ).to(dtype=tokens.dtype)
    numerator = F.adaptive_avg_pool2d(
        flat_tokens * flat_valid, (target_h, target_w)
    )
    denominator = F.adaptive_avg_pool2d(
        flat_valid, (target_h, target_w)
    ).clamp_min(1e-8)
    pooled = numerator / denominator
    pooled_valid = denominator > 0
    pooled = pooled.reshape(
        batch * time * hands, dim, target_h * target_w
    ).transpose(1, 2).reshape(batch, time, hands, target_h * target_w, dim)
    pooled_valid = pooled_valid.reshape(
        batch, time, hands, target_h * target_w
    )
    return pooled, pooled_valid


def _normalized_context_latent_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    scale: float = 0.01,
    hand_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Absolute latent reconstruction loss for visible context tokens."""
    if predicted.shape != target.shape:
        raise ValueError("context prediction and target shapes differ")
    effective_scale = max(float(scale), 1e-6)
    error = F.smooth_l1_loss(
        predicted / effective_scale,
        target.detach() / effective_scale,
        reduction="none",
        beta=0.2,
    ).mean(dim=(-1, -2))
    copy_error = F.smooth_l1_loss(
        torch.zeros_like(target),
        target.detach() / effective_scale,
        reduction="none",
        beta=0.2,
    ).mean(dim=(-1, -2))
    if hand_valid is None:
        loss = error.mean()
        copy = copy_error.mean()
    else:
        if hand_valid.shape != predicted.shape[:1] + predicted.shape[2:3]:
            raise ValueError("hand_valid does not match hand dimension")
        mask = hand_valid[:, None, :].to(error.dtype)
        denominator = mask.sum().clamp_min(1.0)
        loss = (error * mask).sum() / denominator
        copy = (copy_error * mask).sum() / denominator
    cosine = F.cosine_similarity(
        predicted, target.detach(), dim=-1
    ).mean()
    return loss, copy, cosine


def _reduce_metrics(metrics: dict[str, torch.Tensor | float], device: str) -> dict[str, float]:
    if not (dist.is_available() and dist.is_initialized()):
        return {key: float(value) for key, value in metrics.items()}
    keys = list(metrics)
    values = torch.tensor(
        [float(metrics[key]) for key in keys], dtype=torch.float32, device=device
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= dist.get_world_size()
    return {key: float(values[index]) for index, key in enumerate(keys)}


def _broadcast_module_state(module: nn.Module) -> None:
    """Make manually gradient-synchronized replicas start from identical state."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    with torch.no_grad():
        for tensor in module.parameters():
            dist.broadcast(tensor.data, src=0)
        for tensor in module.buffers():
            dist.broadcast(tensor.data, src=0)


def _average_metrics(total: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(1, count) for key, value in total.items()}


class EncoderStage:
    def __init__(self, cfg: dict, device: str):
        self.cfg = cfg
        section = cfg["encoder_training"]
        self.section = section
        self.device = device
        self.encoder = _build_encoder(cfg, device)
        self.decoder = TactilePatchDecoder(
            int(cfg["tactile"].get("embed_dim", 384)),
            out_channels=int(cfg["tactile"].get("in_channels", 3)) * 2,
            patch_size=int(cfg["tactile"].get("patch_size", 14)),
        ).to(device)
        use_reconstruction = (
            float(section.get("lambda_reconstruction", 1.0)) > 0
            or float(section.get("lambda_delta", 0.5)) > 0
        )
        self.latent_self_prediction = (
            not use_reconstruction
            and bool(section.get("latent_self_prediction", True))
        )
        self.latent_context_frames = int(
            section.get("latent_context_frames", 4)
        )
        self.latent_mask_mode = str(
            section.get("latent_mask_mode", "none")
        ).lower()
        self.masked_latent_prediction = (
            self.latent_self_prediction
            and self.latent_mask_mode
            in ("future_frames", "frame", "spatial_tokens", "token")
        )
        if self.latent_self_prediction and not self.masked_latent_prediction:
            raise ValueError(
                "the selected Stage-A recipe requires latent_mask_mode="
                "future_frames or spatial_tokens"
            )
        if self.latent_self_prediction:
            self.encoder_target = copy.deepcopy(self.encoder).eval()
            for parameter in self.encoder_target.parameters():
                parameter.requires_grad_(False)
            predictor_kwargs = {
                "tactile_dim": int(cfg["tactile"].get("embed_dim", 384)),
                "hidden_dim": int(section.get("latent_predictor_dim", 384)),
                "depth": int(section.get("latent_predictor_depth", 2)),
                "heads": int(section.get("latent_predictor_heads", 6)),
                "max_time": int(cfg["context"]) + int(cfg["horizon"]),
                "num_hands": 2,
                "drop_path": float(
                    section.get("latent_predictor_drop_path", 0.05)
                ),
            }
            self.latent_spatial_grid = tuple(
                int(x) for x in section.get("latent_spatial_grid", [2, 2])
            )
            predictor_kwargs["spatial_tokens"] = (
                self.latent_spatial_grid[0] * self.latent_spatial_grid[1]
            )
            predictor_kwargs["query_chunk_size"] = int(
                section.get("latent_query_chunk_size", 4)
            )
            self.latent_predictor = MaskedTouchLatentPredictor(
                **predictor_kwargs
            ).to(device)
        else:
            self.encoder_target = None
            self.latent_predictor = None
            self.latent_spatial_grid = None
        embed_dim = int(cfg["tactile"].get("embed_dim", 384))
        self.contact_probe = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Linear(2 * embed_dim, 1),
        ).to(device)
        self.slip_probe = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Linear(2 * embed_dim, 1),
        ).to(device)
        trainable_modules = [self.encoder, self.contact_probe, self.slip_probe]
        if use_reconstruction:
            trainable_modules.append(self.decoder)
        if self.latent_predictor is not None:
            trainable_modules.append(self.latent_predictor)
        self.opt = torch.optim.AdamW(
            [p for module in trainable_modules for p in module.parameters()],
            lr=float(section.get("lr", 1e-4)),
            weight_decay=float(section.get("weight_decay", 0.05)),
        )
        self.distributed = dist.is_available() and dist.is_initialized()
        dtype = str(section.get("amp_dtype", "float16")).lower()
        self.amp_dtype = torch.bfloat16 if dtype in ("bf16", "bfloat16") else torch.float16
        self.use_amp = bool(section.get("amp", True)) and device.startswith("cuda")
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype == torch.float16
        )
        self.image_size = _image_size(cfg)
        tactile_patch = int(cfg["tactile"].get("patch_size", 16))
        self.encoder_grid = (
            self.image_size[0] // tactile_patch,
            self.image_size[1] // tactile_patch,
        )
        self.tactile_delta = bool(cfg["tactile"].get("use_delta", True))
        self.contact_threshold = float(cfg.get("event_contact_threshold", 0.0039))
        self.slip_threshold = float(cfg.get("event_slip_threshold", 0.0057))
        broadcast_modules = [
            self.encoder,
            self.contact_probe,
            self.slip_probe,
        ]
        if use_reconstruction:
            broadcast_modules.append(self.decoder)
        if self.encoder_target is not None:
            broadcast_modules.append(self.encoder_target)
        if self.latent_predictor is not None:
            broadcast_modules.append(self.latent_predictor)
        for module in broadcast_modules:
            _broadcast_module_state(module)

    def _labels(
        self,
        batch: dict,
        hands: torch.Tensor,
        indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source = str(self.cfg.get("event_label_source", "tactile_proxy")).lower()
        if source == "manifest":
            if batch.get("contact") is None or batch.get("slip") is None:
                raise ValueError("manifest contact/slip labels are required but missing")
            idx = torch.as_tensor(indices, device=self.device, dtype=torch.long)
            contact = batch["contact"].to(self.device).float().index_select(1, idx)
            slip = batch["slip"].to(self.device).float().index_select(1, idx)
            _, _, energy = _event_labels(
                hands,
                indices,
                contact_threshold=self.contact_threshold,
                slip_threshold=self.slip_threshold,
            )
            return contact, slip, energy
        return _event_labels(
            hands,
            indices,
            contact_threshold=self.contact_threshold,
            slip_threshold=self.slip_threshold,
        )

    def _sync(self) -> None:
        if not self.distributed:
            return
        parameters = (
            list(self.encoder.parameters())
            + list(self.contact_probe.parameters())
            + list(self.slip_probe.parameters())
        )
        if not self.latent_self_prediction:
            parameters += list(self.decoder.parameters())
        if self.latent_predictor is not None:
            parameters += list(self.latent_predictor.parameters())
        for parameter in parameters:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(dist.get_world_size())

    def _stage_a_token_mask(
        self,
        batch: int,
        frames: int,
        context_frames: int,
        spatial_tokens: int,
    ) -> torch.Tensor:
        """Create a fine-token mask; true entries replace tokens by mask_token."""
        mask = torch.zeros(
            batch,
            frames,
            2,
            spatial_tokens,
            device=self.device,
            dtype=torch.bool,
        )
        if context_frames >= frames:
            return mask
        if self.latent_mask_mode in ("future_frames", "frame"):
            mask[:, context_frames:] = True
            return mask
        ratio = float(self.section.get("latent_token_mask_ratio", 0.75))
        if not 0.0 < ratio < 1.0:
            raise ValueError("latent_token_mask_ratio must be in (0, 1)")
        generator = None
        if not self.encoder.training:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(self.cfg.get("seed", 0)) + 104729)
        random_values = torch.rand(
            batch,
            frames - context_frames,
            2,
            spatial_tokens,
            device=self.device,
            generator=generator,
        )
        mask[:, context_frames:] = random_values < ratio
        # A completely masked frame cannot contribute any visible coarse cell.
        mask[:, context_frames:, :, 0] = False
        return mask

    def step(self, batch: dict, train: bool) -> dict[str, float]:
        left = _image_tensor_to_float(
            batch["tactile_left"].to(self.device, non_blocking=True)
        )
        right = _image_tensor_to_float(
            batch["tactile_right"].to(self.device, non_blocking=True)
        )
        if left.shape != right.shape:
            raise ValueError("left/right tactile shapes must match")
        hands = torch.stack((left, right), dim=2)
        hand_valid = batch.get("hand_valid")
        if hand_valid is None:
            hand_valid = torch.ones(
                left.shape[0], 2, device=self.device, dtype=torch.bool
            )
        else:
            hand_valid = hand_valid.to(device=self.device, dtype=torch.bool)
        if hand_valid.shape != (left.shape[0], 2):
            raise ValueError("hand_valid must be [B,2]")
        if not hand_valid.any(dim=1).all():
            raise ValueError("each Stage-A sample must contain one valid hand")
        valid = [
            int(index) for index in self.section.get("frame_offsets", [0, 1, 2, 3])
            if int(index) < hands.shape[1]
        ]
        if not valid:
            raise ValueError("Stage A frame_offsets do not fit the loaded window")
        with torch.autocast(
            device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
        ):
            use_reconstruction = (
                float(self.section.get("lambda_reconstruction", 1.0)) > 0
                or float(self.section.get("lambda_delta", 0.5)) > 0
            )
            context_count = min(self.latent_context_frames, len(valid) - 1)
            context_indices = valid[:context_count]
            latent_offsets = [
                int(x)
                for x in self.section.get(
                    "latent_future_offsets",
                    range(1, int(self.cfg["horizon"]) + 1),
                )
            ]
            target_indices = [
                context_count + offset - 1 for offset in latent_offsets
            ]
            if not context_indices:
                raise ValueError("Stage A requires at least one context frame")
            if not target_indices:
                raise ValueError(
                    "latent self-prediction requires future frame offsets"
                )
            if max(target_indices) >= hands.shape[1]:
                raise ValueError(
                    "Stage A future target exceeds the loaded window"
                )
            latent_frame_indices = list(valid)
            if self.masked_latent_prediction:
                all_indices = context_indices + target_indices
                if len(set(all_indices)) != len(all_indices):
                    raise ValueError(
                        "context and future Stage-A indices must not overlap"
                    )
                fine_mask = self._stage_a_token_mask(
                    left.shape[0],
                    len(all_indices),
                    context_count,
                    self.encoder.max_spatial_tokens,
                )
                fine_mask = fine_mask | (
                    ~hand_valid[:, None, :, None]
                )
                token_grid = _encode_causal_indices(
                    self.encoder,
                    left,
                    right,
                    all_indices,
                    self.image_size,
                    use_delta=self.tactile_delta,
                    token_mask=fine_mask,
                )
                with torch.no_grad():
                    target_grid = _encode_causal_indices(
                        self.encoder_target,
                        left,
                        right,
                        all_indices,
                        self.image_size,
                        use_delta=self.tactile_delta,
                    )
                online_grid, online_valid = _pool_spatial_grid_masked(
                    token_grid,
                    ~fine_mask,
                    self.encoder_grid,
                    self.latent_spatial_grid,
                )
                target_grid, _ = _pool_spatial_grid_masked(
                    target_grid,
                    torch.ones(
                        target_grid.shape[:-1],
                        device=target_grid.device,
                        dtype=torch.bool,
                    ),
                    self.encoder_grid,
                    self.latent_spatial_grid,
                )
                prediction = self.latent_predictor(
                    online_grid,
                    memory_mask=online_valid,
                    frame_indices=all_indices,
                    context_indices=context_indices,
                    future_indices=target_indices,
                    base_tokens=online_grid[:, context_count - 1],
                )
                context_prediction = prediction["context_tactile"]
                future_prediction = prediction["future_tactile"]
                context_target = target_grid[:, :context_count]
                future_target = target_grid[:, context_count:]
                (
                    context_loss,
                    context_copy,
                    context_cosine,
                ) = _normalized_context_latent_loss(
                    context_prediction,
                    context_target,
                    scale=float(
                        self.section.get("latent_context_loss_scale", 0.01)
                    ),
                    hand_valid=hand_valid,
                )
                (
                    latent_prediction_loss,
                    latent_prediction_copy,
                    latent_prediction_rms,
                ) = _dual_tactile_delta_loss(
                    future_prediction,
                    future_target,
                    online_grid[:, context_count - 1],
                    scale=float(
                        self.section.get("latent_delta_scale", 0.01)
                    ),
                    direction_weight=float(
                        self.section.get("latent_direction_weight", 0.0)
                    ),
                    per_dim=bool(
                        self.section.get("latent_delta_loss_per_dim", False)
                    ),
                    hand_valid=hand_valid,
                )
                current_grid = online_grid[:, context_count - 1][:, None]
                predict_cosine = F.cosine_similarity(
                    future_prediction - current_grid,
                    future_target - current_grid,
                    dim=-1,
                ).mean()
                loss_current = token_grid.sum() * 0.0
                loss_delta = token_grid.sum() * 0.0
                probe_frame_indices = all_indices
            else:
                pair, target = _causal_tactile_pair(
                    hands, valid, self.image_size
                )
                batch_size, time, hand_count, channels, height, width = pair.shape
                flat = pair.reshape(
                    batch_size * time * hand_count,
                    channels,
                    height,
                    width,
                )
                target_flat = target.reshape_as(flat)
                tokens = self.encoder(flat)
                current_channels = channels // 2
                if use_reconstruction:
                    recon = self.decoder(tokens, self.image_size).squeeze(1)
                    loss_current = F.smooth_l1_loss(
                        recon[:, :current_channels],
                        target_flat[:, :current_channels],
                    )
                    loss_delta = F.smooth_l1_loss(
                        recon[:, current_channels:],
                        target_flat[:, current_channels:],
                    )
                else:
                    loss_current = tokens.sum() * 0.0
                    loss_delta = tokens.sum() * 0.0
                token_grid = tokens.squeeze(1).reshape(
                    batch_size,
                    time,
                    hand_count,
                    tokens.shape[-2],
                    tokens.shape[-1],
                )
                token_view_for_latent = token_grid
                context_pooled = torch.cat(
                    (
                        token_view_for_latent[:, :context_count]
                        .mean(dim=-2)
                        .unsqueeze(-2),
                        token_view_for_latent[:, :context_count]
                        .amax(dim=-2)
                        .unsqueeze(-2),
                    ),
                    dim=-2,
                )
                with torch.no_grad():
                    target_tokens = _encode_indices(
                        self.encoder_target,
                        left,
                        right,
                        target_indices,
                        self.tactile_delta,
                    )
                    target_pooled = target_tokens.mean(dim=-2)
                prediction = self.latent_predictor(
                    context_pooled,
                    context_indices=context_indices,
                    target_indices=target_indices,
                )["future_tactile"]
                current_latent = context_pooled[:, -1, :, :1]
                (
                    latent_prediction_loss,
                    latent_prediction_copy,
                    latent_prediction_rms,
                ) = _dual_tactile_delta_loss(
                    prediction,
                    target_pooled.unsqueeze(-2),
                    current_latent,
                    scale=float(
                        self.section.get("latent_delta_scale", 0.01)
                    ),
                    per_dim=bool(
                        self.section.get("latent_delta_loss_per_dim", False)
                    ),
                    hand_valid=hand_valid,
                )
                context_loss = latent_prediction_loss.sum() * 0.0
                context_copy = context_loss.detach()
                context_cosine = latent_prediction_loss.detach() * 0.0
                predict_cosine = F.cosine_similarity(
                    prediction - current_latent,
                    target_pooled.unsqueeze(-2) - current_latent,
                    dim=-1,
                ).mean()
                probe_frame_indices = valid
            loss_variance = _variance_loss(
                token_grid, target_std=0.1, hand_valid=hand_valid
            )
            loss_covariance = _covariance_loss(
                token_grid, hand_valid=hand_valid
            )
            eps = 1e-6
            latent_relative = (
                float(latent_prediction_loss.detach()) + eps
            ) / (float(latent_prediction_copy.detach()) + eps)
            context_relative = (
                float(context_loss.detach()) + eps
            ) / (float(context_copy.detach()) + eps)
            context_weight = float(
                self.section.get("lambda_context_prediction", 0.5)
            )
            predict_weight = float(
                self.section.get("lambda_latent_prediction", 1.0)
            )
            joint_relative = (
                context_weight * context_relative
                + predict_weight * latent_relative
            ) / max(context_weight + predict_weight, eps)
            frame_positions = {
                int(index): position
                for position, index in enumerate(probe_frame_indices)
            }
            probe_pairs = [
                (frame_positions[int(index)], int(index))
                for index in valid
                if int(index) > 0 and int(index) in frame_positions
            ]
            probe_positions = [pair[0] for pair in probe_pairs]
            probe_indices = [pair[1] for pair in probe_pairs]
            eval_predictions = {}
            if probe_indices:
                contact_target, slip_target, probe_energy = self._labels(
                    batch, hands, probe_indices
                )
                pooled = torch.cat(
                    (
                        token_grid.mean(dim=-2),
                        token_grid.amax(dim=-2),
                    ),
                    dim=-1,
                )
                selected = pooled[:, probe_positions]
                if bool(self.section.get("detach_probe_aux", False)):
                    selected = selected.detach()
                contact_logits = self.contact_probe(selected).squeeze(-1)
                slip_logits = self.slip_probe(selected).squeeze(-1)
                hand_mask = hand_valid[:, None, :].expand_as(contact_target)
                contact_error = F.binary_cross_entropy_with_logits(
                    contact_logits, contact_target, reduction="none"
                )
                slip_error = F.binary_cross_entropy_with_logits(
                    slip_logits, slip_target, reduction="none"
                )
                loss_contact = contact_error[hand_mask].mean()
                loss_slip = slip_error[hand_mask].mean()
                contact_auroc = _auroc(
                    contact_logits[hand_mask], contact_target[hand_mask]
                )
                slip_auroc = _auroc(
                    slip_logits[hand_mask], slip_target[hand_mask]
                )
                eval_predictions = {
                    "contact_logits": contact_logits[hand_mask].detach(),
                    "contact_target": contact_target[hand_mask].detach(),
                    "slip_logits": slip_logits[hand_mask].detach(),
                    "slip_target": slip_target[hand_mask].detach(),
                }
                pair_positions = []
                pair_indices = []
                for position, index in enumerate(probe_frame_indices):
                    previous = int(index) - 1
                    if previous in frame_positions:
                        pair_positions.append(
                            (position, frame_positions[previous])
                        )
                        pair_indices.append(int(index))
                if pair_positions:
                    current_positions = [pair[0] for pair in pair_positions]
                    previous_positions = [pair[1] for pair in pair_positions]
                    current_features = F.normalize(pooled[:, current_positions], dim=-1)
                    previous_features = F.normalize(pooled[:, previous_positions], dim=-1)
                    feature_change = (
                        1.0 - (current_features * previous_features).sum(dim=-1)
                    ).clamp_min(0.0)
                    _, _, pair_energy = self._labels(batch, hands, pair_indices)
                    target_change = (
                        float(self.section.get("temporal_change_scale", 0.05))
                        * (pair_energy / self.contact_threshold).clamp(0.0, 2.0)
                    ).detach()
                    temporal_error = F.smooth_l1_loss(
                        feature_change, target_change, reduction="none"
                    )
                    temporal_mask = hand_valid[:, None, :].expand_as(
                        feature_change
                    )
                    loss_temporal = temporal_error[temporal_mask].mean()
                else:
                    loss_temporal = token_grid.sum() * 0.0
            else:
                loss_contact = loss_slip = token_grid.sum() * 0.0
                loss_temporal = token_grid.sum() * 0.0
                contact_auroc = slip_auroc = 0.5
            loss = (
                float(self.section.get("lambda_reconstruction", 1.0)) * loss_current
                + float(self.section.get("lambda_delta", 0.5)) * loss_delta
                + float(self.section.get("lambda_variance", 0.05)) * loss_variance
                + float(self.section.get("lambda_covariance", 0.01)) * loss_covariance
                + float(self.section.get("lambda_contact", 0.05)) * loss_contact
                + float(self.section.get("lambda_slip", 0.05)) * loss_slip
                + float(self.section.get("lambda_temporal", 0.2)) * loss_temporal
                + float(
                    self.section.get("lambda_context_prediction", 0.5)
                )
                * context_loss
                + float(self.section.get("lambda_latent_prediction", 1.0))
                * latent_prediction_loss
            )
        metrics = {
            "loss": loss.detach(),
            "loss_current": loss_current.detach(),
            "loss_delta": loss_delta.detach(),
            "loss_variance": loss_variance.detach(),
            "loss_covariance": loss_covariance.detach(),
            "loss_contact": loss_contact.detach(),
            "loss_slip": loss_slip.detach(),
            "loss_temporal": loss_temporal.detach(),
            "loss_context_prediction": context_loss.detach(),
            "loss_context_prediction_copy": context_copy.detach(),
            "context_relative": torch.tensor(
                context_relative, device=self.device
            ),
            "context_cosine": context_cosine.detach(),
            "predict_cosine": predict_cosine.detach(),
            "latent_joint_relative": torch.tensor(
                joint_relative, device=self.device
            ),
            "loss_latent_prediction": latent_prediction_loss.detach(),
            "loss_latent_prediction_copy": latent_prediction_copy.detach(),
            "latent_prediction_rms": latent_prediction_rms.detach(),
            "latent_relative": torch.tensor(
                latent_relative, device=self.device
            ),
            "contact_auroc": torch.tensor(contact_auroc, device=self.device),
            "slip_auroc": torch.tensor(slip_auroc, device=self.device),
            "hand_valid_left": hand_valid[:, 0].float().mean().detach(),
            "hand_valid_right": hand_valid[:, 1].float().mean().detach(),
            "single_hand_rate": (
                hand_valid.sum(dim=1) == 1
            ).float().mean().detach(),
            "dual_hand_rate": (
                hand_valid.sum(dim=1) == 2
            ).float().mean().detach(),
        }
        if not train:
            self._eval_predictions = eval_predictions
            return {key: float(value) for key, value in metrics.items()}
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self._sync()
        self.scaler.unscale_(self.opt)
        clip_parameters = (
            list(self.encoder.parameters())
            + list(self.contact_probe.parameters())
            + list(self.slip_probe.parameters())
        )
        if not self.latent_self_prediction:
            clip_parameters += list(self.decoder.parameters())
        if self.latent_predictor is not None:
            clip_parameters += list(self.latent_predictor.parameters())
        torch.nn.utils.clip_grad_norm_(clip_parameters, 1.0)
        self.scaler.step(self.opt)
        self.scaler.update()
        if self.encoder_target is not None:
            update_ema(
                self.encoder,
                self.encoder_target,
                float(self.section.get("ema_encoder", 0.999)),
            )
        return {key: float(value) for key, value in metrics.items()}

    def state_dict(self) -> dict:
        state = {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "contact_probe": self.contact_probe.state_dict(),
            "slip_probe": self.slip_probe.state_dict(),
            "encoder_target": self.encoder_target.state_dict()
            if self.encoder_target is not None else None,
            "latent_predictor": self.latent_predictor.state_dict()
            if self.latent_predictor is not None else None,
            "optimizer": self.opt.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
        }
        return state


def _evaluate_encoder(
    trainer: EncoderStage,
    loader: DataLoader | None,
    max_batches: int,
) -> dict[str, float]:
    if loader is None:
        return {}
    trainer.encoder.eval()
    trainer.decoder.eval()
    trainer.contact_probe.eval()
    trainer.slip_probe.eval()
    if trainer.latent_predictor is not None:
        trainer.latent_predictor.eval()
    total, count = {}, 0
    predictions = {}
    with torch.no_grad():
        for batch in loader:
            metrics = trainer.step(batch, train=False)
            for key, value in metrics.items():
                total[key] = total.get(key, 0.0) + value
            for key, value in getattr(trainer, "_eval_predictions", {}).items():
                predictions.setdefault(key, []).append(value)
            count += 1
            if max_batches and count >= max_batches:
                break
    trainer.encoder.train()
    trainer.decoder.train()
    trainer.contact_probe.train()
    trainer.slip_probe.train()
    if trainer.latent_predictor is not None:
        trainer.latent_predictor.train()
    averaged = _average_metrics(total, count)
    if predictions:
        contact_logits = torch.cat(predictions["contact_logits"])
        contact_target = torch.cat(predictions["contact_target"])
        slip_logits = torch.cat(predictions["slip_logits"])
        slip_target = torch.cat(predictions["slip_target"])
        averaged["contact_auroc"] = _auroc(contact_logits, contact_target)
        averaged["slip_auroc"] = _auroc(slip_logits, slip_target)
    return averaged


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_encoder_stage(
    cfg: dict,
    *,
    device: str,
    rank: int,
    world_size: int,
    output: str,
    resume: str | None,
    max_steps: int | None,
    max_records: int | None,
) -> Path:
    section = cfg["encoder_training"]
    trainer = EncoderStage(cfg, device)
    batch_size = int(section.get("batch_size", cfg.get("batch_size", 4)))
    train_loader, val_loader, test_loader, split = _build_loaders(
        cfg,
        stage="encoder",
        batch_size=batch_size,
        distributed=world_size > 1,
        rank=rank,
        world_size=world_size,
        max_records=max_records,
    )
    max_epochs = section.get("max_epochs")
    if max_steps is None and max_epochs is not None:
        steps = max(1, int(math.ceil(len(train_loader) * float(max_epochs))))
    else:
        steps = int(max_steps or section.get("max_steps", 2000))
    start_step = 0
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        resume_optimizer = True
        saved_spatial = state.get("encoder", {}).get("spatial_pos")
        if (
            saved_spatial is not None
            and saved_spatial.shape != trainer.encoder.spatial_pos.shape
        ):
            resume_optimizer = False
        _load_encoder_payload(trainer.encoder, state["encoder"])
        if (
            trainer.encoder_target is not None
            and state.get("encoder_target") is not None
        ):
            trainer.encoder_target.load_state_dict(state["encoder_target"])
        if (
            trainer.latent_predictor is not None
            and state.get("latent_predictor") is not None
        ):
            trainer.latent_predictor.load_state_dict(state["latent_predictor"])
        try:
            trainer.decoder.load_state_dict(state["decoder"])
        except RuntimeError:
            print(
                "warning: Stage A decoder shape changed; reinitializing decoder",
                flush=True,
            )
        if state.get("contact_probe") is not None:
            trainer.contact_probe.load_state_dict(state["contact_probe"])
            trainer.slip_probe.load_state_dict(state["slip_probe"])
        if (
            resume_optimizer
            and state.get("optimizer") is not None
            and state.get("contact_probe") is not None
        ):
            trainer.opt.load_state_dict(state["optimizer"])
        if state.get("scaler") is not None and trainer.scaler.is_enabled():
            trainer.scaler.load_state_dict(state["scaler"])
        start_step = int(state.get("step", 0))
        if start_step >= steps:
            raise ValueError(
                f"resume step {start_step} is not below requested max_steps={steps}"
            )
    scheduler = WarmupCosineMultiplier(
        trainer.opt,
        total_steps=steps,
        warmup_steps=int(section.get("warmup_steps", 100)),
        min_lr_scale=float(section.get("min_lr_scale", 0.05)),
    )
    scheduler.jump_to(start_step)
    output_path = Path(output)
    if is_main_process():
        output_path.mkdir(parents=True, exist_ok=True)
        _write_json(output_path / "split.json", split)
        (output_path / "resolved_config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
        )
    writer = None
    if is_main_process() and bool(section.get("tensorboard", True)):
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(output_path / "tensorboard"))
    best_score = float("inf")
    best_step = None
    bad_checks = 0
    records = {}
    global_step = start_step
    epoch = start_step // max(1, len(train_loader))
    log_every = max(1, int(section.get("log_every", 20)))
    check_every = max(1, int(section.get("checkpoint_every_steps", 250)))
    min_steps = int(section.get("early_stopping_min_steps", 0))
    patience = int(section.get("early_stopping_patience", 0))
    min_delta = float(section.get("early_stopping_min_delta", 0.0))
    while global_step < steps:
        should_stop = False
        epoch_steps = 0
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        for batch in train_loader:
            epoch_steps += 1
            global_step += 1
            metrics = _reduce_metrics(trainer.step(batch, train=True), device)
            scheduler.step()
            if is_main_process() and (global_step == 1 or global_step % log_every == 0):
                print(
                    f"[encoder] step {global_step}/{steps} "
                    + ", ".join(f"{key}={value:.5f}" for key, value in metrics.items()),
                    flush=True,
                )
                if writer is not None:
                    for key, value in metrics.items():
                        writer.add_scalar(f"{key}/train_step", value, global_step)
            if global_step % check_every == 0 or global_step >= steps:
                val = {}
                should_stop = False
                if is_main_process():
                    val = _evaluate_encoder(
                        trainer, val_loader,
                        int(section.get("max_val_batches", 24)),
                    )
                    reconstruction_active = (
                        float(section.get("lambda_reconstruction", 1.0)) > 0
                        or float(section.get("lambda_delta", 0.5)) > 0
                    )
                    if reconstruction_active:
                        tactile_loss = (
                            float(val.get("loss_current", 0.0))
                            + float(section.get("lambda_delta", 0.5))
                            * float(val.get("loss_delta", 0.0))
                        )
                        selection_rule = (
                            "loss_current + lambda_delta * loss_delta "
                            "+ probe_selection_weight * "
                            "(2 - contact_auroc - slip_auroc)"
                        )
                    else:
                        relative_key = (
                            "latent_joint_relative"
                            if trainer.masked_latent_prediction
                            else "latent_relative"
                        )
                        tactile_loss = float(
                            val.get(relative_key, val.get("loss_temporal", 0.0))
                        )
                        selection_rule = (
                            f"{relative_key} + probe_selection_weight * "
                            "(2 - contact_auroc - slip_auroc)"
                        )
                    semantic_penalty = (
                        1.0 - float(val.get("contact_auroc", 0.5))
                        + 1.0 - float(val.get("slip_auroc", 0.5))
                    )
                    score = (
                        tactile_loss
                        + float(section.get("probe_selection_weight", 0.0005))
                        * semantic_penalty
                    )
                    payload = {
                        "step": global_step,
                        "score": score,
                        "validation": val,
                        **trainer.state_dict(),
                    }
                    latest = output_path / "latest_encoder.pt"
                    best = output_path / "best_encoder.pt"
                    torch.save(payload, latest)
                    improved = score < best_score - min_delta
                    if improved or best_step is None:
                        best_score = score
                        best_step = global_step
                        bad_checks = 0
                        torch.save(payload, best)
                    elif global_step >= min_steps:
                        bad_checks += 1
                    should_stop = (
                        global_step >= min_steps
                        and patience > 0
                        and bad_checks >= patience
                    )
                    records[str(global_step)] = {
                        "score": score,
                        "tactile_loss": tactile_loss,
                        "validation": val,
                        "best_step": best_step,
                        "bad_checks": bad_checks,
                    }
                    _write_json(
                        output_path / "metrics.json",
                        {
                            "best_step": best_step,
                            "best_score": best_score,
                            "selection_rule": selection_rule,
                            "records": records,
                        },
                    )
                    print(
                        f"[encoder] validation step={global_step} score={score:.6f} "
                        + ", ".join(f"{key}={value:.5f}" for key, value in val.items()),
                        flush=True,
                    )
                    if writer is not None:
                        for key, value in val.items():
                            writer.add_scalar(f"{key}/val_step", value, global_step)
                if world_size > 1:
                    stop_tensor = torch.tensor(
                        [int(should_stop)], dtype=torch.int32, device=device
                    )
                    dist.broadcast(stop_tensor, src=0)
                    should_stop = bool(stop_tensor.item())
                if should_stop:
                    if is_main_process():
                        print(
                            f"[encoder] early stopping at step {global_step}",
                            flush=True,
                        )
                    break
                if world_size > 1:
                    distributed_barrier()
            if global_step >= steps:
                break
        if should_stop:
            break
        if epoch_steps == 0:
            break
        epoch += 1
    if writer is not None:
        writer.close()
    if is_main_process() and test_loader is not None:
        best_path = output_path / "best_encoder.pt"
        if best_path.is_file():
            best_state = torch.load(best_path, map_location=device, weights_only=False)
            trainer.encoder.load_state_dict(best_state["encoder"])
            if (
                trainer.encoder_target is not None
                and best_state.get("encoder_target") is not None
            ):
                trainer.encoder_target.load_state_dict(
                    best_state["encoder_target"]
                )
            if (
                trainer.latent_predictor is not None
                and best_state.get("latent_predictor") is not None
            ):
                trainer.latent_predictor.load_state_dict(
                    best_state["latent_predictor"]
                )
            trainer.decoder.load_state_dict(best_state["decoder"])
            if best_state.get("contact_probe") is not None:
                trainer.contact_probe.load_state_dict(best_state["contact_probe"])
                trainer.slip_probe.load_state_dict(best_state["slip_probe"])
        test_metrics = _evaluate_encoder(
            trainer, test_loader,
            int(
                section.get(
                    "max_test_batches",
                    section.get("max_val_batches", 24),
                )
            ),
        )
        _write_json(
            output_path / "test_metrics.json",
            {
                "test_metrics": test_metrics,
                "split": {
                    "n_test_records": split["n_test_records"],
                    "n_test_windows": split["n_test_windows"],
                    "test_groups": split["test_groups"],
                },
            },
        )
    if world_size > 1:
        distributed_barrier()
    return output_path / "best_encoder.pt"


def _load_frozen_encoder(path: str | Path, cfg: dict, device: str) -> TactileEncoder:
    encoder = _build_encoder(cfg, device)
    state = torch.load(path, map_location=device, weights_only=False)
    payload = state.get("encoder", state)
    _load_encoder_payload(encoder, payload)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def _load_encoder_payload(encoder: TactileEncoder, payload: dict) -> None:
    current = encoder.state_dict()
    compatible = {}
    skipped = []
    for key, value in payload.items():
        if key in current and current[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)
    encoder.load_state_dict(compatible, strict=False)
    if skipped:
        print(
            f"warning: encoder initialization skipped {len(skipped)} incompatible "
            f"tensor(s): {skipped[:8]}",
            flush=True,
        )


def _encode_indices(
    encoder: TactileEncoder,
    left: torch.Tensor,
    right: torch.Tensor,
    indices: list[int],
    use_delta: bool,
) -> torch.Tensor:
    parts = [
        _encode_tactile_indices(encoder, hand, indices, causal_delta=use_delta)
        for hand in (left, right)
    ]
    return torch.stack(parts, dim=2)


def _dual_tactile_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
    alpha: float = 1.0,
    hand_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = F.normalize(predicted, dim=-1)
    tgt = F.normalize(target, dim=-1)
    error = (pred - tgt).square().mean(dim=-1)
    reference = F.normalize(current, dim=-1).unsqueeze(1)
    change = 1.0 - (tgt * reference).sum(dim=-1).clamp(-1.0, 1.0)
    if hand_valid is None:
        mean_change = change.mean(dim=(2, 3), keepdim=True)
        mask = torch.ones_like(change)
    else:
        if hand_valid.shape != change.shape[:1] + change.shape[2:3]:
            raise ValueError("hand_valid does not match hand dimension")
        mask = hand_valid[:, None, :, None].expand_as(change).to(change.dtype)
        mean_change = (
            change * mask
        ).sum(dim=(2, 3), keepdim=True) / mask.sum(
            dim=(2, 3), keepdim=True
        ).clamp_min(1.0)
    relative = change / mean_change.clamp_min(1e-6)
    weights = (1.0 + float(alpha) * relative).detach()
    weighted_mask = weights * mask
    loss = (error * weighted_mask).sum() / weighted_mask.sum().clamp_min(1e-6)
    copy = (
        F.normalize(current, dim=-1).unsqueeze(1).expand_as(predicted)
        - tgt
    ).square().mean(dim=-1)
    copy_loss = (copy * weighted_mask).sum() / weighted_mask.sum().clamp_min(1e-6)
    return loss, copy_loss


def _summary_4way(features: torch.Tensor) -> torch.Tensor:
    """Summarize ``[B,T,H,N,D]`` touch latents with four time/space statistics."""
    if features.ndim != 5:
        raise ValueError("summary features must be [B,T,H,N,D]")
    spatial_mean = features.mean(dim=-2)
    spatial_max = features.amax(dim=-2)
    return torch.stack(
        (
            spatial_mean.mean(dim=1),
            spatial_mean.amax(dim=1),
            spatial_max.mean(dim=1),
            spatial_max.amax(dim=1),
        ),
        dim=2,
    )


def _temporal_2way(features: torch.Tensor) -> torch.Tensor:
    """Summarize a ``[B,T,D]`` sequence with temporal mean and max."""
    if features.ndim != 3:
        raise ValueError("temporal summary features must be [B,T,D]")
    return torch.stack((features.mean(dim=1), features.amax(dim=1)), dim=1)


def _normalized_summary_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    hand_valid: torch.Tensor | None = None,
    hand_dim: int = 1,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"condition summary shape mismatch: {tuple(prediction.shape)} "
            f"!= {tuple(target.shape)}"
        )
    error = (
        F.normalize(prediction, dim=-1) - F.normalize(target, dim=-1)
    ).square().mean(dim=-1)
    if hand_valid is None:
        return error.mean()
    shape = [1] * error.ndim
    shape[0] = hand_valid.shape[0]
    shape[hand_dim] = hand_valid.shape[1]
    mask = hand_valid.view(*shape).expand_as(error).to(error.dtype)
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


def _token_diversity_loss(tokens: torch.Tensor) -> torch.Tensor:
    """Penalize redundant condition slots and expose their cosine diversity."""
    if tokens.shape[1] < 2:
        return tokens.sum() * 0.0
    normalized = F.normalize(tokens, dim=-1)
    gram = normalized @ normalized.transpose(1, 2)
    eye = torch.eye(
        tokens.shape[1], device=tokens.device, dtype=torch.bool
    )[None]
    return gram.masked_select(~eye).square().mean()


def _dual_tactile_delta_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
    scale: float = 0.01,
    alpha: float = 1.0,
    direction_weight: float = 0.1,
    per_dim: bool = False,
    hand_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict future tactile change instead of nearly time-invariant tokens."""
    base = current.unsqueeze(1)
    predicted_delta = predicted - base
    target_delta = target - base
    magnitude = target_delta.square().mean(dim=-1).sqrt()
    if hand_valid is None:
        mask = torch.ones_like(magnitude)
        mean_magnitude = magnitude.mean(dim=(2, 3), keepdim=True)
    else:
        if hand_valid.shape != magnitude.shape[:1] + magnitude.shape[2:3]:
            raise ValueError("hand_valid does not match hand dimension")
        mask = hand_valid[:, None, :, None].expand_as(magnitude).to(magnitude.dtype)
        mean_magnitude = (
            magnitude * mask
        ).sum(dim=(2, 3), keepdim=True) / mask.sum(
            dim=(2, 3), keepdim=True
        ).clamp_min(1.0)
    weights = 1.0 + float(alpha) * (
        magnitude / mean_magnitude.clamp_min(1e-6)
    ).detach()
    weighted_mask = weights * mask
    scale = max(float(scale), 1e-6)
    if per_dim:
        global_scale = (
            target_delta.square().mean().sqrt().clamp_min(scale * 0.25)
        ).detach()
        dim_scale = (
            target_delta.square()
            .mean(dim=(0, 1, 2, 3), keepdim=True)
            .sqrt()
            .clamp_min(scale * 0.1)
            .detach()
        )
        effective_scale = torch.maximum(
            dim_scale, global_scale * 0.1
        )
    else:
        effective_scale = scale
    error = F.smooth_l1_loss(
        predicted_delta / effective_scale,
        target_delta / effective_scale,
        reduction="none",
        beta=0.2,
    ).mean(dim=-1)
    loss = (error * weighted_mask).sum() / weighted_mask.sum().clamp_min(1e-6)
    zero = F.smooth_l1_loss(
        torch.zeros_like(target_delta),
        target_delta / effective_scale,
        reduction="none",
        beta=0.2,
    ).mean(dim=-1)
    zero = (zero * weighted_mask).sum() / weighted_mask.sum().clamp_min(1e-6)
    direction_error = (
        F.normalize(predicted_delta, dim=-1)
        - F.normalize(target_delta, dim=-1)
    ).square().mean(dim=-1)
    copy_direction_error = (
        F.normalize(torch.zeros_like(target_delta), dim=-1)
        - F.normalize(target_delta, dim=-1)
    ).square().mean(dim=-1)
    direction_loss = (
        direction_error * weighted_mask
    ).sum() / weighted_mask.sum().clamp_min(1e-6)
    copy_direction = (
        copy_direction_error * weighted_mask
    ).sum() / weighted_mask.sum().clamp_min(1e-6)
    loss = loss + float(direction_weight) * direction_loss
    zero = zero + float(direction_weight) * copy_direction
    return loss, zero, target_delta.square().mean().sqrt()


def _dense_tactile_relative(
    event_relative: float,
    planning_relative: float,
    event_count: int,
    planning_count: int,
) -> float:
    total = max(1, int(event_count) + int(planning_count))
    return (
        int(event_count) * float(event_relative)
        + int(planning_count) * float(planning_relative)
    ) / total


def _predictor_selection_loss(
    section: dict,
    *,
    dense: bool,
    tactile_relative: float,
    condition_loss: float,
    visual_relative: float,
    contact_auroc: float,
    slip_auroc: float,
) -> float:
    """Apply one selection objective in both training and evaluation."""
    if dense:
        return (
            float(section.get("selection_tactile_weight", 0.75))
            * float(tactile_relative)
            + float(section.get("selection_condition_weight", 0.10))
            * float(condition_loss)
            + float(section.get("selection_visual_weight", 0.0))
            * float(visual_relative)
            + float(section.get("selection_contact_weight", 0.075))
            * (1.0 - float(contact_auroc))
            + float(section.get("selection_slip_weight", 0.075))
            * (1.0 - float(slip_auroc))
        )
    tactile_weight = (
        float(section.get("lambda_event_tactile", 1.0))
        + float(section.get("lambda_planning_tactile", 1.0))
    )
    if tactile_weight > 0:
        return (
            0.75 * float(tactile_relative)
            + 0.125 * (1.0 - float(contact_auroc))
            + 0.125 * (1.0 - float(slip_auroc))
        )
    return (
        0.5 * (1.0 - float(contact_auroc))
        + 0.5 * (1.0 - float(slip_auroc))
    )


def _event_labels(
    tactile: torch.Tensor,
    event_indices: list[int],
    *,
    contact_threshold: float = 0.0039,
    slip_threshold: float = 0.0057,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Proxy contact/slip labels from tactile deformation.

    Contact uses deviation from the per-window temporal-median sensor image;
    slip uses frame-to-frame deformation energy. Thresholds are fixed from a
    deterministic calibration sample.
    """
    if tactile.shape[1] <= max(event_indices):
        raise ValueError("event index exceeds tactile window")
    gray = tactile.mean(dim=3)
    baseline = gray.median(dim=1, keepdim=True).values
    future_idx = torch.as_tensor(event_indices, device=tactile.device, dtype=torch.long)
    previous = gray.index_select(1, future_idx - 1)
    current = gray.index_select(1, future_idx)
    contact_energy = (current - baseline).abs().mean(dim=(-1, -2))
    slip_energy = (current - previous).abs().mean(dim=(-1, -2))
    contact = (contact_energy > float(contact_threshold)).float()
    slip = (slip_energy > float(slip_threshold)).float()
    return contact, slip, slip_energy


def _auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.detach().float().reshape(-1)
    labels = labels.detach().float().reshape(-1) > 0.5
    positive = scores[labels]
    negative = scores[~labels]
    if positive.numel() == 0 or negative.numel() == 0:
        return 0.5
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    _, inverse, counts = torch.unique_consecutive(
        sorted_scores, return_inverse=True, return_counts=True
    )
    starts = counts.cumsum(dim=0) - counts
    average_ranks = (starts + 1 + starts + counts).to(scores.dtype) / 2
    ranks = average_ranks[inverse]
    positive_rank_sum = ranks[labels[order]].sum()
    n_positive = positive.numel()
    n_negative = negative.numel()
    statistic = positive_rank_sum - n_positive * (n_positive + 1) / 2
    return float(statistic / (n_positive * n_negative))


class PredictorStage:
    def __init__(self, cfg: dict, section: dict, device: str, encoder_path: str):
        self.cfg = cfg
        self.section = section
        self.device = device
        tactile_dim = int(cfg["tactile"].get("embed_dim", 384))
        self.encoder = _load_frozen_encoder(encoder_path, cfg, device)
        self.fine_tune_encoder = bool(section.get("fine_tune_encoder", False))
        if self.fine_tune_encoder:
            self.encoder_target = copy.deepcopy(self.encoder).eval()
            for parameter in self.encoder_target.parameters():
                parameter.requires_grad_(False)
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(True)
            self.encoder.train()
        else:
            self.encoder_target = self.encoder
        self.contact_probe = nn.Sequential(
            nn.LayerNorm(2 * tactile_dim),
            nn.Linear(2 * tactile_dim, 1),
        ).to(device)
        self.slip_probe = nn.Sequential(
            nn.LayerNorm(2 * tactile_dim),
            nn.Linear(2 * tactile_dim, 1),
        ).to(device)
        encoder_state = torch.load(
            encoder_path, map_location=device, weights_only=False
        )
        if (
            self.fine_tune_encoder
            and encoder_state.get("encoder_target") is not None
        ):
            try:
                self.encoder_target.load_state_dict(
                    encoder_state["encoder_target"]
                )
            except RuntimeError:
                print(
                    "warning: Stage-A EMA target is incompatible; "
                    "using online encoder initialization",
                    flush=True,
                )
        if encoder_state.get("contact_probe") is not None:
            try:
                self.contact_probe.load_state_dict(encoder_state["contact_probe"])
                self.slip_probe.load_state_dict(encoder_state["slip_probe"])
            except RuntimeError:
                print(
                    "warning: Stage A probe shape changed; reinitializing probes",
                    flush=True,
                )
        self.visual = _build_visual(cfg, section, device)
        visual_dim = int(getattr(
            self.visual, "embed_dim",
            getattr(getattr(self.visual, "backbone", None), "embed_dim", 768),
        ))
        tactile_dim = int(cfg["tactile"].get("embed_dim", 384))
        image_size = _image_size(cfg)
        patch = int(cfg["tactile"].get("patch_size", 14))
        crop = int(cfg.get("crop_size", 384))
        visual_patch = int(cfg.get("model", {}).get("patch_size", 16))
        tactile_tokens = (image_size[0] // patch) * (image_size[1] // patch)
        self.pool_tactile_spatial = bool(
            section.get("pool_tactile_spatial", True)
        )
        self.context_pooling = str(
            cfg.get("context_pooling", "mean")
        ).lower()
        if self.context_pooling not in ("mean", "mean_max", "spatial_grid"):
            raise ValueError(
                "context_pooling must be 'mean', 'mean_max', or 'spatial_grid'"
            )
        self.tactile_source_grid = (
            image_size[0] // patch,
            image_size[1] // patch,
        )
        self.visual_source_grid = (
            crop // visual_patch,
            crop // visual_patch,
        )
        self.tactile_context_grid = tuple(
            int(x) for x in section.get("tactile_context_grid", [2, 2])
        )
        self.visual_context_grid = tuple(
            int(x) for x in section.get("visual_context_grid", [4, 4])
        )
        if self.pool_tactile_spatial:
            tactile_tokens = 1
        architecture = str(section.get("architecture", "dense")).lower()
        if architecture not in ("dense", "dense30"):
            raise ValueError(
                "the selected Stage-B recipe requires architecture=dense"
            )
        self.dense = True
        self.predictor = DenseTactilePredictor(
            tactile_dim=tactile_dim,
            visual_dim=visual_dim,
            state_dim=int(
                section.get(
                    "state_history_dim",
                    cfg.get(
                        "state_history_dim",
                        2 * int(cfg.get("action_dim", 8)),
                    ),
                )
            ),
            hidden_dim=int(section.get("predictor_dim", 384)),
            depth=int(section.get("predictor_depth", 2)),
            heads=int(section.get("predictor_heads", 6)),
            max_time=int(cfg["context"]) + int(cfg["horizon"]) + 1,
            num_hands=2,
            future_offsets=cfg.get(
                "future_offsets", range(1, int(cfg["horizon"]) + 1)
            ),
            visual_future_offsets=cfg.get(
                "visual_future_offsets",
                range(2, int(cfg["horizon"]) + 1, 2),
            ),
            num_condition_tokens=int(
                section.get("num_condition_tokens", 4)
            ),
            drop_path=float(section.get("drop_path", 0.05)),
        ).to(device)
        self.predictor.detach_auxiliary_paths = bool(
            section.get("detach_auxiliary_paths", False)
        )
        if bool(section.get("train_auxiliary_only", False)):
            for name, parameter in self.predictor.named_parameters():
                parameter.requires_grad_(
                    name.startswith("future_visual_head.")
                    or name.startswith("condition_")
                )
        source_forecaster = encoder_state.get("latent_predictor")
        if (
            bool(section.get("init_from_latent_predictor", True))
            and source_forecaster is not None
        ):
            source = source_forecaster
            target = self.predictor.state_dict()
            warm = {}
            for key, value in source.items():
                mapped_key = "tactile_in" if key == "input_proj.weight" else key
                if key == "input_proj.bias":
                    mapped_key = "tactile_in.bias"
                if key == "time_embedding":
                    count = min(value.shape[1], target[mapped_key].shape[1])
                    target[mapped_key][:, :count].copy_(value[:, :count])
                    continue
                if (
                    mapped_key in target
                    and target[mapped_key].shape == value.shape
                ):
                    warm[mapped_key] = value
            self.predictor.load_state_dict(warm, strict=False)
            print(
                "initialized dense predictor from Stage-A latent predictor "
                f"({len(warm)} tensors)",
                flush=True,
            )
        self.freeze_predictor = bool(section.get("freeze_predictor", False))
        if self.freeze_predictor:
            self.predictor.eval()
            for parameter in self.predictor.parameters():
                parameter.requires_grad_(False)
        groups = [{
            "params": list(self.contact_probe.parameters())
            + list(self.slip_probe.parameters()),
            "lr": float(section.get("lr_probe", section.get("lr", 2e-4))),
        }]
        if not self.freeze_predictor:
            groups.insert(0, {
                "params": self.predictor.parameters(),
                "lr": float(section.get("lr", 2e-4)),
            })
        if self.fine_tune_encoder:
            groups.append({
                "params": self.encoder.parameters(),
                "lr": float(section.get("lr_encoder", 1e-5)),
            })
        self.opt = torch.optim.AdamW(
            groups,
            weight_decay=float(section.get("weight_decay", 0.05)),
        )
        self.distributed = dist.is_available() and dist.is_initialized()
        dtype = str(section.get("amp_dtype", "float16")).lower()
        self.amp_dtype = torch.bfloat16 if dtype in ("bf16", "bfloat16") else torch.float16
        self.use_amp = bool(section.get("amp", True)) and device.startswith("cuda")
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype == torch.float16
        )
        self.context = int(cfg["context"])
        self.tubelet = int(cfg.get("visual_tubelet_size", 2))
        self.predict_visual = bool(cfg.get("predict_visual", True))
        self.event_offsets = tuple(int(x) for x in cfg["event_offsets"])
        self.planning_offsets = tuple(int(x) for x in cfg["planning_offsets"])
        self.tactile_delta = bool(cfg["tactile"].get("use_delta", True))
        self.contact_threshold = float(cfg.get("event_contact_threshold", 0.0039))
        self.slip_threshold = float(cfg.get("event_slip_threshold", 0.0057))
        self.action_dim = cfg.get("action_dim")
        self.state_history_dim = int(
            section.get(
                "state_history_dim",
                cfg.get("state_history_dim", 2 * int(self.action_dim or 1)),
            )
        )
        self.state_scale = torch.as_tensor(
            cfg.get("state_scale", [1.0] * int(self.action_dim or 1)),
            dtype=torch.float32,
            device=device,
        ).view(1, 1, -1)
        self.modality_warmup_steps = int(
            section.get("modality_warmup_steps", 0)
        )
        if self.dense and self.modality_warmup_steps > 0:
            self.predictor.use_visual = False
            self.predictor.use_state = False
        for module in (
            self.encoder,
            self.encoder_target,
            self.contact_probe,
            self.slip_probe,
            self.predictor,
        ):
            _broadcast_module_state(module)

    def _sync(self) -> None:
        if not self.distributed:
            return
        for parameter in self.predictor.parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(dist.get_world_size())
        if self.fine_tune_encoder:
            for parameter in self.encoder.parameters():
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                parameter.grad.div_(dist.get_world_size())
        for parameter in (
            list(self.contact_probe.parameters()) + list(self.slip_probe.parameters())
        ):
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(dist.get_world_size())

    def _visual_targets(
        self, video: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if video.ndim != 5:
            raise ValueError("vision must be [B,T,C,H,W] before V-JEPA")
        video = video.permute(0, 2, 1, 3, 4).contiguous()
        total_steps = video.shape[2] // self.tubelet
        context_steps = self.context // self.tubelet
        with torch.no_grad():
            context = _visual_tokens(
                _run_visual(self.visual, video[:, :, :self.context]),
                video.shape[0], context_steps,
            )
            if not self.predict_visual:
                return context, None
            future = _visual_tokens(
                _run_visual(self.visual, video), video.shape[0], total_steps
            )
        visual_idx = [
            context_steps + int(offset) // self.tubelet - 1
            for offset in self.cfg.get(
                "visual_future_offsets",
                self.planning_offsets,
            )
        ]
        visual_offsets = tuple(
            int(x)
            for x in self.cfg.get(
                "visual_future_offsets",
                self.planning_offsets,
            )
        )
        if any(offset % self.tubelet for offset in visual_offsets):
            raise ValueError("planning offsets must align to the visual tubelet grid")
        return context, future[:, visual_idx]

    def _state_history(self, batch: dict, frames: int) -> torch.Tensor:
        state = batch.get("state")
        if state is None:
            raise ValueError("historical state is required")
        state = state.to(self.device, non_blocking=True).float()[:, :frames]
        if state.shape[-1] != self.state_scale.shape[-1]:
            raise ValueError(
                f"state width {state.shape[-1]} != configured "
                f"{self.state_scale.shape[-1]}"
            )
        normalized = state / self.state_scale
        delta = torch.zeros_like(normalized)
        delta[:, 1:] = normalized[:, 1:] - normalized[:, :-1]
        pieces = [normalized]
        if bool(self.cfg.get("state_history_include_delta", True)):
            pieces.append(delta)
        if bool(self.cfg.get("state_history_include_labels", False)):
            for key in ("contact", "slip"):
                labels = batch.get(key)
                if labels is None:
                    raise ValueError(f"{key} labels are required in state history")
                pieces.append(
                    labels.to(self.device, non_blocking=True).float()[:, :frames]
                )
        state_history = torch.cat(pieces, dim=-1)
        if state_history.shape[-1] != self.state_history_dim:
            raise ValueError(
                f"state history width {state_history.shape[-1]} != "
                f"configured {self.state_history_dim}"
            )
        return state_history

    def _labels(
        self,
        batch: dict,
        tactile: torch.Tensor,
        indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source = str(self.cfg.get("event_label_source", "tactile_proxy")).lower()
        if source == "manifest":
            if batch.get("contact") is None or batch.get("slip") is None:
                raise ValueError("manifest contact/slip labels are required but missing")
            idx = torch.as_tensor(indices, device=self.device, dtype=torch.long)
            contact = batch["contact"].to(self.device).float().index_select(1, idx)
            slip = batch["slip"].to(self.device).float().index_select(1, idx)
            if contact.ndim == 2:
                contact = contact[:, :, None].expand(-1, -1, 2)
                slip = slip[:, :, None].expand(-1, -1, 2)
            _, _, energy = _event_labels(
                tactile,
                indices,
                contact_threshold=self.contact_threshold,
                slip_threshold=self.slip_threshold,
            )
            return contact, slip, energy
        return _event_labels(
            tactile,
            indices,
            contact_threshold=self.contact_threshold,
            slip_threshold=self.slip_threshold,
        )

    def step(self, batch: dict, train: bool) -> dict[str, float]:
        left = _image_tensor_to_float(
            batch["tactile_left"].to(self.device, non_blocking=True)
        )
        right = _image_tensor_to_float(
            batch["tactile_right"].to(self.device, non_blocking=True)
        )
        video = _prepare_visual_gpu(
            batch["vision"].to(self.device, non_blocking=True),
            int(self.cfg.get("crop_size", 384)),
        )
        state_history = self._state_history(batch, self.context)
        tactile = torch.stack((left, right), dim=2)
        hand_valid = batch.get("hand_valid")
        if hand_valid is None:
            hand_valid = torch.ones(
                left.shape[0], 2, device=self.device, dtype=torch.bool
            )
        else:
            hand_valid = hand_valid.to(device=self.device, dtype=torch.bool)
        if hand_valid.shape != (left.shape[0], 2):
            raise ValueError("hand_valid must be [B,2]")
        if not hand_valid.any(dim=1).all():
            raise ValueError("each Stage-B sample must contain one valid hand")
        context_indices = list(range(self.context))
        event_indices = [self.context + offset - 1 for offset in self.event_offsets]
        planning_indices = [self.context + offset - 1 for offset in self.planning_offsets]
        future_offsets = tuple(int(x) for x in self.cfg.get(
            "future_offsets",
            tuple(self.event_offsets) + tuple(self.planning_offsets),
        ))
        future_indices = [self.context + offset - 1 for offset in future_offsets]
        with torch.autocast(
            device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
        ):
            tactile_context = _encode_indices(
                self.encoder, left, right, context_indices, self.tactile_delta
            )
            tactile_context_full = tactile_context
            with torch.no_grad():
                if self.dense:
                    future_target = _encode_indices(
                        self.encoder_target,
                        left,
                        right,
                        future_indices,
                        self.tactile_delta,
                    )
                    split = len(self.event_offsets)
                    event_target = future_target[:, :split]
                    planning_target = future_target[:, split:]
                    future_target_dense_full = future_target
                    future_target_dense_raw = future_target
                else:
                    future_target_dense_full = None
                    future_target_dense_raw = None
                    event_target = _encode_indices(
                        self.encoder_target, left, right, event_indices, self.tactile_delta
                    )
                    planning_target = _encode_indices(
                        self.encoder_target, left, right, planning_indices, self.tactile_delta
                    )
                visual_context, visual_target = self._visual_targets(video)
            visual_context_full = visual_context
            if self.pool_tactile_spatial:
                tactile_mean = tactile_context.mean(dim=-2, keepdim=True)
                tactile_parts = [tactile_mean]
                if self.context_pooling in ("mean_max", "spatial_grid"):
                    tactile_parts.append(
                        tactile_context.amax(dim=-2, keepdim=True)
                    )
                if self.context_pooling == "spatial_grid":
                    tactile_parts.append(
                        _pool_spatial_grid(
                            tactile_context,
                            self.tactile_source_grid,
                            self.tactile_context_grid,
                        )
                    )
                tactile_context = torch.cat(tactile_parts, dim=-2)
                event_target = event_target.mean(dim=-2, keepdim=True)
                planning_target = planning_target.mean(dim=-2, keepdim=True)
                if future_target_dense_full is not None:
                    future_target_dense_full = (
                        future_target_dense_full.mean(dim=-2, keepdim=True)
                    )
            if self.dense:
                visual_mean = visual_context.mean(dim=-2, keepdim=True)
                visual_parts = [visual_mean]
                if self.context_pooling in ("mean_max", "spatial_grid"):
                    visual_parts.append(
                        visual_context.amax(dim=-2, keepdim=True)
                    )
                if self.context_pooling == "spatial_grid":
                    visual_parts.append(
                        _pool_spatial_grid(
                            visual_context,
                            self.visual_source_grid,
                            self.visual_context_grid,
                        )
                    )
                visual_context = torch.cat(visual_parts, dim=-2)
            output = self.predictor(
                visual_context,
                tactile_context,
                visual_indices=[
                    self.tubelet - 1 + step * self.tubelet
                    for step in range(self.context // self.tubelet)
                ],
                tactile_indices=context_indices,
                context_frames=self.context,
                state_history=state_history,
                hand_valid=hand_valid,
            )
            current_tactile = (
                tactile_context[:, -1, :, :1]
                if tactile_context.ndim == 5
                else tactile_context[:, -1].unsqueeze(-2)
            )
            if self.dense:
                split = len(self.event_offsets)
                event_prediction = output["future_tactile"][:, :split]
                planning_prediction = output["future_tactile"][:, split:]
            else:
                event_prediction = output["event_tactile"]
                planning_prediction = output["planning_tactile"]
            event_loss, event_copy = _dual_tactile_loss(
                event_prediction,
                event_target,
                current_tactile,
                hand_valid=hand_valid,
            )
            planning_loss, planning_copy = _dual_tactile_loss(
                planning_prediction,
                planning_target,
                current_tactile,
                hand_valid=hand_valid,
            )
            delta_scale = float(self.section.get("delta_loss_scale", 0.01))
            event_delta_loss, event_delta_copy, event_delta_rms = (
                _dual_tactile_delta_loss(
                    event_prediction, event_target, current_tactile,
                    scale=delta_scale,
                    direction_weight=float(
                        self.section.get("delta_direction_weight", 0.1)
                    ),
                    per_dim=bool(
                        self.section.get("delta_loss_per_dim", False)
                    ),
                    hand_valid=hand_valid,
                )
            )
            planning_delta_loss, planning_delta_copy, planning_delta_rms = (
                _dual_tactile_delta_loss(
                    planning_prediction, planning_target, current_tactile,
                    scale=delta_scale,
                    direction_weight=float(
                        self.section.get("delta_direction_weight", 0.1)
                    ),
                    per_dim=bool(
                        self.section.get("delta_loss_per_dim", False)
                    ),
                    hand_valid=hand_valid,
                )
            )
            if visual_target is None:
                visual_anchor = (
                    output["future_visual"]
                    if "future_visual" in output
                    else output["future_tactile"]
                )
                visual_loss = visual_anchor.sum() * 0.0
                visual_copy = visual_loss.detach()
                visual_delta_loss = visual_loss.detach()
                visual_delta_copy = visual_loss.detach()
                visual_delta_rms = visual_loss.detach()
                visual_absolute_loss = visual_loss.detach()
                visual_absolute_copy = visual_loss.detach()
            elif self.dense:
                visual_prediction = output["future_visual"]
                visual_loss_target = visual_target.mean(dim=-2)
                visual_reference = visual_context_full[:, -1].mean(
                    dim=-2, keepdim=True
                )
                visual_loss = F.mse_loss(
                    F.normalize(visual_prediction, dim=-1),
                    F.normalize(visual_loss_target, dim=-1),
                )
                visual_copy = F.mse_loss(
                    F.normalize(visual_reference.expand_as(visual_loss_target), dim=-1),
                    F.normalize(visual_loss_target, dim=-1),
                )
                visual_delta_weight = float(
                    self.section.get("visual_delta_weight", 1.0)
                )
                visual_reference_dense = visual_reference.expand_as(
                    visual_loss_target
                )
                visual_target_delta = (
                    visual_loss_target - visual_reference_dense
                )
                visual_delta_rms = (
                    visual_target_delta.square().mean().sqrt()
                    .clamp_min(1e-6).detach()
                )
                visual_delta_loss = F.smooth_l1_loss(
                    (visual_prediction - visual_reference_dense)
                    / visual_delta_rms,
                    visual_target_delta / visual_delta_rms,
                    beta=0.2,
                )
                visual_delta_copy = F.smooth_l1_loss(
                    torch.zeros_like(visual_target_delta),
                    visual_target_delta / visual_delta_rms,
                    beta=0.2,
                )
                visual_absolute_loss = visual_loss
                visual_absolute_copy = visual_copy
                visual_loss = (
                    visual_absolute_loss
                    + visual_delta_weight * visual_delta_loss
                )
                visual_copy = (
                    visual_absolute_copy
                    + visual_delta_weight * visual_delta_copy
                )
            condition_summary_loss = (
                output["future_tactile"].sum() * 0.0
                if self.dense else
                output["event_tactile"].sum() * 0.0
            )
            condition_history_loss = condition_summary_loss
            condition_future_tactile_loss = condition_summary_loss
            condition_history_visual_loss = condition_summary_loss
            condition_future_visual_loss = condition_summary_loss
            condition_history_state_loss = condition_summary_loss
            condition_future_trajectory_loss = condition_summary_loss
            if self.dense:
                history_condition_target = _summary_4way(
                    tactile_context_full
                )
                future_condition_target = _summary_4way(
                    future_target_dense_raw
                )
                history_visual_condition_target = _temporal_2way(
                    visual_context_full.mean(dim=-2)
                )
                condition_history_loss = _normalized_summary_loss(
                    output["condition_history_summary"],
                    history_condition_target,
                    hand_valid=hand_valid,
                    hand_dim=1,
                )
                condition_future_tactile_loss = _normalized_summary_loss(
                    output["condition_future_tactile_summary"],
                    future_condition_target,
                    hand_valid=hand_valid,
                    hand_dim=1,
                )
                future_trajectory_target = future_target_dense_full.mean(dim=-2)
                condition_future_trajectory_loss = _normalized_summary_loss(
                    output["condition_future_tactile_trajectory"],
                    future_trajectory_target,
                    hand_valid=hand_valid,
                    hand_dim=2,
                )
                condition_history_visual_loss = _normalized_summary_loss(
                    output["condition_history_visual_summary"],
                    history_visual_condition_target,
                )
                if visual_target is not None:
                    future_visual_condition_target = _temporal_2way(
                        visual_target.mean(dim=-2)
                    )
                    condition_future_visual_loss = _normalized_summary_loss(
                        output["condition_future_visual_summary"],
                        future_visual_condition_target,
                    )
                condition_history_state_loss = _normalized_summary_loss(
                    output["condition_history_state_summary"],
                    _temporal_2way(state_history),
                )
                condition_summary_loss = (
                    condition_history_loss
                    + condition_future_tactile_loss
                    + condition_history_visual_loss
                    + condition_future_visual_loss
                    + condition_history_state_loss
                )
                condition_diversity_loss = _token_diversity_loss(
                    output["condition_tokens"]
                )
            else:
                condition_diversity_loss = condition_summary_loss
            contact_target, slip_target, event_energy = self._labels(
                batch, tactile, event_indices
            )
            if self.dense:
                contact_loss = event_prediction.sum() * 0.0
                slip_loss = event_prediction.sum() * 0.0
                contact_auroc = slip_auroc = 0.5
            else:
                contact_loss = F.binary_cross_entropy_with_logits(
                    output["event_contact_logits"], contact_target
                )
                slip_loss = F.binary_cross_entropy_with_logits(
                    output["event_slip_logits"], slip_target
                )
                contact_auroc = _auroc(
                    output["event_contact_logits"], contact_target
                )
                slip_auroc = _auroc(
                    output["event_slip_logits"], slip_target
                )
            current_contact_target, current_slip_target, _ = self._labels(
                batch, tactile, [self.context - 1]
            )
            current_tokens = tactile_context_full[:, -1]
            if bool(self.section.get("detach_current_probe", False)):
                current_tokens = current_tokens.detach()
            probe_features = torch.cat(
                (current_tokens.mean(dim=-2), current_tokens.amax(dim=-2)), dim=-1
            )
            current_contact_logits = self.contact_probe(probe_features).squeeze(-1)
            current_slip_logits = self.slip_probe(probe_features).squeeze(-1)
            current_contact = current_contact_target[:, 0]
            current_slip = current_slip_target[:, 0]
            probe_contact_error = F.binary_cross_entropy_with_logits(
                current_contact_logits,
                current_contact,
                pos_weight=torch.tensor(
                    float(self.section.get("probe_contact_pos_weight", 1.0)),
                    device=self.device,
                ),
                reduction="none",
            )
            probe_slip_error = F.binary_cross_entropy_with_logits(
                current_slip_logits,
                current_slip,
                pos_weight=torch.tensor(
                    float(self.section.get("probe_slip_pos_weight", 3.0)),
                    device=self.device,
                ),
                reduction="none",
            )
            probe_contact_loss = probe_contact_error[hand_valid].mean()
            probe_slip_loss = probe_slip_error[hand_valid].mean()
            if self.dense:
                magnitude = event_prediction.sum() * 0.0
                future_delta_loss = (
                    len(self.event_offsets) * event_delta_loss
                    + len(self.planning_offsets) * planning_delta_loss
                ) / (
                    len(self.event_offsets) + len(self.planning_offsets)
                )
                tactile_loss_term = future_delta_loss
            else:
                magnitude = F.smooth_l1_loss(
                    output["event_tactile"].norm(dim=-1),
                    event_target.norm(dim=-1),
                )
                tactile_loss_term = (
                    float(self.section.get("lambda_event_tactile", 1.0))
                    * event_delta_loss
                    + float(self.section.get("lambda_planning_tactile", 1.0))
                    * planning_delta_loss
                )
            loss = (
                tactile_loss_term
                + float(self.section.get("lambda_planning_visual", 0.5)) * visual_loss
                + float(self.section.get("lambda_contact", 0.1)) * contact_loss
                + float(self.section.get("lambda_slip", 0.1)) * slip_loss
                + float(self.section.get("lambda_magnitude", 0.1)) * magnitude
                + float(self.section.get("lambda_probe_contact", 0.2))
                * probe_contact_loss
                + float(self.section.get("lambda_probe_slip", 0.2))
                * probe_slip_loss
                + float(self.section.get("lambda_condition", 0.1))
                * condition_summary_loss
                + float(self.section.get("lambda_condition_trajectory", 0.0))
                * condition_future_trajectory_loss
                + float(self.section.get("lambda_condition_diversity", 0.05))
                * condition_diversity_loss
            )
        condition_tokens = (
            output["condition_tokens"]
            if "condition_tokens" in output
            else None
        )
        metrics = {
            "loss": loss.detach(),
            "loss_event_tactile": event_loss.detach(),
            "loss_event_copy_last": event_copy.detach(),
            "loss_event_delta": event_delta_loss.detach(),
            "loss_event_delta_copy": event_delta_copy.detach(),
            "event_delta_rms": event_delta_rms.detach(),
            "loss_planning_tactile": planning_loss.detach(),
            "loss_planning_copy_last": planning_copy.detach(),
            "loss_planning_delta": planning_delta_loss.detach(),
            "loss_planning_delta_copy": planning_delta_copy.detach(),
            "planning_delta_rms": planning_delta_rms.detach(),
            "loss_future_delta": (
                future_delta_loss.detach()
                if self.dense else
                (event_delta_loss + planning_delta_loss).detach()
            ),
            "loss_planning_visual": visual_loss.detach(),
            "loss_visual_absolute": visual_absolute_loss.detach(),
            "loss_visual_absolute_copy_last": visual_absolute_copy.detach(),
            "loss_visual_copy_last": visual_copy.detach(),
            "loss_visual_delta": visual_delta_loss.detach(),
            "loss_visual_delta_copy": visual_delta_copy.detach(),
            "visual_delta_rms": visual_delta_rms.detach(),
            "visual_delta_relative": (
                visual_delta_loss.detach()
                / visual_delta_copy.detach().clamp_min(1e-6)
            ),
            "visual_relative": (
                visual_loss.detach() / visual_copy.detach().clamp_min(1e-6)
            ),
            "loss_contact": contact_loss.detach(),
            "loss_slip": slip_loss.detach(),
            "loss_probe_contact": probe_contact_loss.detach(),
            "loss_probe_slip": probe_slip_loss.detach(),
            "loss_magnitude": magnitude.detach(),
            "loss_condition": condition_summary_loss.detach(),
            "loss_condition_diversity": condition_diversity_loss.detach(),
            "loss_condition_history": condition_history_loss.detach(),
            "loss_condition_future_tactile": (
                condition_future_tactile_loss.detach()
            ),
            "loss_condition_future_trajectory": (
                condition_future_trajectory_loss.detach()
            ),
            "loss_condition_history_visual": (
                condition_history_visual_loss.detach()
            ),
            "loss_condition_future_visual": (
                condition_future_visual_loss.detach()
            ),
            "loss_condition_history_state": (
                condition_history_state_loss.detach()
            ),
            "condition_token_norm": (
                condition_tokens.norm(dim=-1).mean().detach()
                if condition_tokens is not None
                else loss.detach() * 0.0
            ),
            "condition_token_std": (
                condition_tokens.std(dim=1).mean().detach()
                if condition_tokens is not None
                else loss.detach() * 0.0
            ),
            "condition_token_diversity": (
                1.0 - condition_diversity_loss.detach()
                if condition_tokens is not None
                else loss.detach() * 0.0
            ),
            "visual_gate": torch.tanh(self.predictor.visual_gate).detach()
            if self.dense else loss.detach() * 0.0,
            "state_gate": torch.tanh(self.predictor.state_gate).detach()
            if self.dense else loss.detach() * 0.0,
            "contact_positive_rate": contact_target.float().mean().detach(),
            "slip_positive_rate": slip_target.float().mean().detach(),
            "event_energy": event_energy.float().mean().detach(),
            "contact_auroc": torch.tensor(contact_auroc, device=self.device),
            "slip_auroc": torch.tensor(slip_auroc, device=self.device),
            "probe_contact_auroc": torch.tensor(
                _auroc(
                    current_contact_logits[hand_valid],
                    current_contact[hand_valid],
                ),
                device=self.device,
            ),
            "probe_slip_auroc": torch.tensor(
                _auroc(
                    current_slip_logits[hand_valid],
                    current_slip[hand_valid],
                ),
                device=self.device,
            ),
            "hand_valid_left": hand_valid[:, 0].float().mean().detach(),
            "hand_valid_right": hand_valid[:, 1].float().mean().detach(),
            "single_hand_rate": (
                hand_valid.sum(dim=1) == 1
            ).float().mean().detach(),
            "dual_hand_rate": (
                hand_valid.sum(dim=1) == 2
            ).float().mean().detach(),
        }
        ratio_eps = 1e-6
        event_relative = (
            float(event_delta_loss.detach()) + ratio_eps
        ) / (float(event_delta_copy.detach()) + ratio_eps)
        planning_relative = (
            float(planning_delta_loss.detach()) + ratio_eps
        ) / (float(planning_delta_copy.detach()) + ratio_eps)
        if self.dense:
            tactile_relative = _dense_tactile_relative(
                event_relative,
                planning_relative,
                len(self.event_offsets),
                len(self.planning_offsets),
            )
        else:
            tactile_relative = 0.5 * (event_relative + planning_relative)
        contact = float(metrics["probe_contact_auroc"])
        slip = float(metrics["probe_slip_auroc"])
        selection_loss = _predictor_selection_loss(
            self.section,
            dense=self.dense,
            tactile_relative=tactile_relative,
            condition_loss=(
                float(condition_summary_loss.detach())
                + float(condition_future_trajectory_loss.detach())
            ),
            visual_relative=float(metrics["visual_relative"]),
            contact_auroc=contact,
            slip_auroc=slip,
        )
        metrics.update({
            "event_relative": torch.tensor(event_relative, device=self.device),
            "planning_relative": torch.tensor(planning_relative, device=self.device),
            "tactile_relative": torch.tensor(tactile_relative, device=self.device),
            "selection_loss": torch.tensor(selection_loss, device=self.device),
        })
        if not train:
            predictions = {
                "probe_contact_logits": current_contact_logits[hand_valid].detach(),
                "probe_contact_target": current_contact[hand_valid].detach(),
                "probe_slip_logits": current_slip_logits[hand_valid].detach(),
                "probe_slip_target": current_slip[hand_valid].detach(),
            }
            if not self.dense:
                predictions.update({
                    "contact_logits": output["event_contact_logits"].detach(),
                    "contact_target": contact_target.detach(),
                    "slip_logits": output["event_slip_logits"].detach(),
                    "slip_target": slip_target.detach(),
                })
            if self.dense and visual_target is not None:
                predictions.update({
                    "visual_prediction": visual_prediction.detach(),
                    "visual_target": visual_loss_target.detach(),
                    "visual_reference": visual_reference.detach()[:, 0],
                })
            if self.dense:
                predictions.update({
                    "condition_tokens": output["condition_tokens"].detach(),
                    "condition_history_prediction": (
                        output["condition_history_summary"].detach()
                    ),
                    "condition_history_target": history_condition_target.detach(),
                    "condition_future_tactile_prediction": (
                        output["condition_future_tactile_summary"].detach()
                    ),
                    "condition_future_tactile_target": (
                        future_condition_target.detach()
                    ),
                    "condition_future_trajectory_prediction": (
                        output["condition_future_tactile_trajectory"].detach()
                    ),
                    "condition_future_trajectory_target": (
                        future_trajectory_target.detach()
                    ),
                    "condition_history_visual_prediction": (
                        output["condition_history_visual_summary"].detach()
                    ),
                    "condition_history_visual_target": (
                        history_visual_condition_target.detach()
                    ),
                    "condition_history_state_prediction": (
                        output["condition_history_state_summary"].detach()
                    ),
                    "condition_history_state_target": (
                        _temporal_2way(state_history).detach()
                    ),
                })
                if visual_target is not None:
                    predictions.update({
                        "condition_future_visual_prediction": (
                            output["condition_future_visual_summary"].detach()
                        ),
                        "condition_future_visual_target": (
                            future_visual_condition_target.detach()
                        )
                    })
            self._eval_predictions = predictions
            return {key: float(value) for key, value in metrics.items()}
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self._sync()
        self.scaler.unscale_(self.opt)
        clip_parameters = list(self.predictor.parameters())
        clip_parameters += list(self.contact_probe.parameters())
        clip_parameters += list(self.slip_probe.parameters())
        if self.fine_tune_encoder:
            clip_parameters += list(self.encoder.parameters())
        torch.nn.utils.clip_grad_norm_(clip_parameters, 1.0)
        self.scaler.step(self.opt)
        self.scaler.update()
        if (
            self.fine_tune_encoder
            and float(self.section.get("lr_encoder", 1e-5)) > 0
        ):
            update_ema(
                self.encoder,
                self.encoder_target,
                float(self.section.get("ema_encoder", 0.999)),
            )
        return {key: float(value) for key, value in metrics.items()}

    def state_dict(self) -> dict:
        return {
            "predictor": self.predictor.state_dict(),
            "contact_probe": self.contact_probe.state_dict(),
            "slip_probe": self.slip_probe.state_dict(),
            "encoder": self.encoder.state_dict(),
            "encoder_target": self.encoder_target.state_dict()
            if self.fine_tune_encoder else None,
            "optimizer": self.opt.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
        }


def _evaluate_predictor(
    trainer: PredictorStage,
    loader: DataLoader | None,
    max_batches: int,
) -> dict[str, float]:
    if loader is None:
        return {}
    trainer.predictor.eval()
    trainer.contact_probe.eval()
    trainer.slip_probe.eval()
    if trainer.fine_tune_encoder:
        trainer.encoder.eval()
    total, count = {}, 0
    predictions = {}
    with torch.no_grad():
        for batch in loader:
            metrics = trainer.step(batch, train=False)
            for key, value in metrics.items():
                total[key] = total.get(key, 0.0) + value
            for key, value in getattr(trainer, "_eval_predictions", {}).items():
                predictions.setdefault(key, []).append(value)
            count += 1
            if max_batches and count >= max_batches:
                break
    if not trainer.freeze_predictor:
        trainer.predictor.train()
    trainer.contact_probe.train()
    trainer.slip_probe.train()
    if trainer.fine_tune_encoder:
        trainer.encoder.train()
    averaged = _average_metrics(total, count)
    if predictions:
        for prefix in ("contact", "slip", "probe_contact", "probe_slip"):
            logits_key = f"{prefix}_logits"
            target_key = f"{prefix}_target"
            if logits_key not in predictions or target_key not in predictions:
                continue
            logits = torch.cat(predictions[logits_key])
            target = torch.cat(predictions[target_key])
            averaged[f"{prefix}_auroc"] = _auroc(logits, target)
        for name in (
            "condition_history",
            "condition_future_tactile",
            "condition_future_trajectory",
            "condition_history_visual",
            "condition_future_visual",
            "condition_history_state",
            "visual",
        ):
            prediction_key = f"{name}_prediction"
            target_key = f"{name}_target"
            if (
                prediction_key not in predictions
                or target_key not in predictions
            ):
                continue
            prediction = torch.cat(predictions[prediction_key]).float()
            target = torch.cat(predictions[target_key]).float()
            averaged[f"{name}_cosine"] = float(
                F.cosine_similarity(prediction, target, dim=-1).mean()
            )
            averaged[f"{name}_normalized_mse"] = float(
                F.mse_loss(
                    F.normalize(prediction, dim=-1),
                    F.normalize(target, dim=-1),
                )
            )
    ratio_eps = 1e-6
    event_relative = (
        averaged.get("loss_event_delta", 0.0) + ratio_eps
    ) / (averaged.get("loss_event_delta_copy", 0.0) + ratio_eps)
    planning_relative = (
        averaged.get("loss_planning_delta", 0.0) + ratio_eps
    ) / (averaged.get("loss_planning_delta_copy", 0.0) + ratio_eps)
    if trainer.dense:
        tactile_relative = _dense_tactile_relative(
            event_relative,
            planning_relative,
            len(trainer.event_offsets),
            len(trainer.planning_offsets),
        )
    else:
        tactile_relative = 0.5 * (event_relative + planning_relative)
    contact = averaged.get("probe_contact_auroc", 0.5)
    slip = averaged.get("probe_slip_auroc", 0.5)
    selection_loss = _predictor_selection_loss(
        trainer.section,
        dense=trainer.dense,
        tactile_relative=tactile_relative,
        condition_loss=(
            averaged.get("loss_condition", 0.0)
            + averaged.get("loss_condition_future_trajectory", 0.0)
        ),
        visual_relative=averaged.get("visual_relative", 0.0),
        contact_auroc=contact,
        slip_auroc=slip,
    )
    averaged.update({
        "event_relative": event_relative,
        "planning_relative": planning_relative,
        "tactile_relative": tactile_relative,
        "selection_loss": selection_loss,
    })
    return averaged


def run_predictor_stage(
    cfg: dict,
    *,
    device: str,
    rank: int,
    world_size: int,
    output: str,
    encoder_path: str,
    resume: str | None,
    max_steps: int | None,
    max_records: int | None,
) -> Path:
    section = cfg["predictor_training"]
    trainer = PredictorStage(cfg, section, device, encoder_path)
    batch_size = int(section.get("batch_size", cfg.get("batch_size", 4)))
    train_loader, val_loader, test_loader, split = _build_loaders(
        cfg,
        stage="predictor",
        batch_size=batch_size,
        distributed=world_size > 1,
        rank=rank,
        world_size=world_size,
        max_records=max_records,
    )
    max_epochs = section.get("max_epochs")
    if max_steps is None and max_epochs is not None:
        steps = max(1, int(math.ceil(len(train_loader) * float(max_epochs))))
    else:
        steps = int(max_steps or section.get("max_steps", 4000))
    start_step = 0
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        trainer.predictor.load_state_dict(state["predictor"])
        if state.get("contact_probe") is not None:
            try:
                trainer.contact_probe.load_state_dict(state["contact_probe"])
                trainer.slip_probe.load_state_dict(state["slip_probe"])
            except RuntimeError:
                print(
                    "warning: checkpoint probe shape changed; reinitializing probes",
                    flush=True,
                )
        if (
            state.get("encoder") is not None
            and bool(section.get("resume_restore_encoder", True))
        ):
            trainer.encoder.load_state_dict(state["encoder"])
        if (
            trainer.fine_tune_encoder
            and state.get("encoder_target") is not None
            and bool(section.get("resume_restore_encoder", True))
        ):
            trainer.encoder_target.load_state_dict(state["encoder_target"])
        if (
            state.get("optimizer") is not None
            and state.get("contact_probe") is not None
            and bool(section.get("resume_restore_optimizer", True))
        ):
            saved_groups = state["optimizer"].get("param_groups", [])
            if len(saved_groups) == len(trainer.opt.param_groups):
                trainer.opt.load_state_dict(state["optimizer"])
            else:
                print(
                    "warning: optimizer parameter groups changed; "
                    "continuing with a fresh optimizer",
                    flush=True,
                )
        if state.get("scaler") is not None and trainer.scaler.is_enabled():
            trainer.scaler.load_state_dict(state["scaler"])
        start_step = int(state.get("step", 0))
        if start_step >= steps:
            raise ValueError(
                f"resume step {start_step} is not below max_steps={steps}"
            )
    scheduler = WarmupCosineMultiplier(
        trainer.opt,
        total_steps=steps,
        warmup_steps=int(section.get("warmup_steps", 200)),
        min_lr_scale=float(section.get("min_lr_scale", 0.05)),
    )
    scheduler.jump_to(start_step)
    output_path = Path(output)
    if is_main_process():
        output_path.mkdir(parents=True, exist_ok=True)
        _write_json(output_path / "split.json", split)
        pseudo_cfg = dict(cfg)
        pseudo_cfg["encoder_checkpoint"] = str(encoder_path)
        (output_path / "resolved_config.yaml").write_text(
            yaml.safe_dump(pseudo_cfg, sort_keys=False), encoding="utf-8"
        )
    writer = None
    if is_main_process() and bool(section.get("tensorboard", True)):
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(output_path / "tensorboard"))
    best_score = float("inf")
    best_step = None
    bad_checks = 0
    records = {}
    global_step = start_step
    epoch = start_step // max(1, len(train_loader))
    log_every = max(1, int(section.get("log_every", 20)))
    check_every = max(1, int(section.get("checkpoint_every_steps", 250)))
    min_steps = int(section.get("early_stopping_min_steps", 2000))
    patience = int(section.get("early_stopping_patience", 5))
    min_delta = float(section.get("early_stopping_min_delta", 0.002))
    while global_step < steps:
        should_stop = False
        epoch_steps = 0
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        for batch in train_loader:
            epoch_steps += 1
            global_step += 1
            if trainer.dense and trainer.modality_warmup_steps > 0:
                modalities_enabled = (
                    global_step >= trainer.modality_warmup_steps
                )
                trainer.predictor.use_visual = modalities_enabled
                trainer.predictor.use_state = modalities_enabled
            metrics = _reduce_metrics(trainer.step(batch, train=True), device)
            scheduler.step()
            if is_main_process() and (global_step == 1 or global_step % log_every == 0):
                print(
                    f"[predictor] step {global_step}/{steps} "
                    + ", ".join(f"{key}={value:.5f}" for key, value in metrics.items()),
                    flush=True,
                )
                if writer is not None:
                    for key, value in metrics.items():
                        writer.add_scalar(f"{key}/train_step", value, global_step)
            should_stop = False
            if global_step % check_every == 0 or global_step >= steps:
                if is_main_process():
                    val = _evaluate_predictor(
                        trainer, val_loader,
                        int(section.get("max_val_batches", 24)),
                    )
                    score = float(val.get("selection_loss", float("inf")))
                    payload = {
                        "step": global_step,
                        "score": score,
                        "validation": val,
                        **trainer.state_dict(),
                    }
                    torch.save(payload, output_path / "latest_predictor.pt")
                    improved = score < best_score - min_delta
                    if improved or best_step is None:
                        best_score = score
                        best_step = global_step
                        bad_checks = 0
                        torch.save(payload, output_path / "best_predictor.pt")
                    elif global_step >= min_steps:
                        bad_checks += 1
                    should_stop = (
                        global_step >= min_steps and patience > 0
                        and bad_checks >= patience
                    )
                    record = {
                        "score": score,
                        "step": global_step,
                        "validation": val,
                        "best_step": best_step,
                        "bad_checks": bad_checks,
                    }
                    records[str(global_step)] = record
                    _write_json(
                        output_path / "metrics.json",
                        {"best_step": best_step, "best_score": best_score,
                         "records": records},
                    )
                    print(
                        f"[predictor] validation step={global_step} score={score:.6f} "
                        f"tactile_rel={val.get('tactile_relative')} "
                        f"probe_contact={val.get('probe_contact_auroc')} "
                        f"probe_slip={val.get('probe_slip_auroc')} "
                        f"event_contact={val.get('contact_auroc')} "
                        f"event_slip={val.get('slip_auroc')} "
                        f"best_step={best_step} bad={bad_checks}/{patience}",
                        flush=True,
                    )
                    if writer is not None:
                        for key, value in val.items():
                            writer.add_scalar(f"{key}/val_step", value, global_step)
                if world_size > 1:
                    stop_tensor = torch.tensor(
                        [int(should_stop)], dtype=torch.int32, device=device
                    )
                    dist.broadcast(stop_tensor, src=0)
                    should_stop = bool(stop_tensor.item())
            if should_stop:
                if is_main_process():
                    print(
                        f"[predictor] early stopping at step {global_step}", flush=True
                    )
                break
            if global_step >= steps:
                break
        if should_stop:
            break
        if epoch_steps == 0:
            break
        epoch += 1
    if writer is not None:
        writer.close()
    if is_main_process() and test_loader is not None:
        best_path = output_path / "best_predictor.pt"
        if best_path.is_file():
            best_state = torch.load(best_path, map_location=device, weights_only=False)
            trainer.predictor.load_state_dict(best_state["predictor"])
            trainer.encoder.load_state_dict(best_state["encoder"])
            if best_state.get("encoder_target") is not None:
                trainer.encoder_target.load_state_dict(best_state["encoder_target"])
            if best_state.get("contact_probe") is not None:
                trainer.contact_probe.load_state_dict(best_state["contact_probe"])
                trainer.slip_probe.load_state_dict(best_state["slip_probe"])
        test_metrics = _evaluate_predictor(
            trainer, test_loader,
            int(
                section.get(
                    "max_test_batches",
                    section.get("max_val_batches", 24),
                )
            ),
        )
        _write_json(
            output_path / "test_metrics.json",
            {
                "test_metrics": test_metrics,
                "split": {
                    "n_test_records": split["n_test_records"],
                    "n_test_windows": split["n_test_windows"],
                    "test_groups": split["test_groups"],
                },
            },
        )
    if world_size > 1:
        distributed_barrier()
    return output_path / "best_predictor.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage", choices=("encoder", "predictor", "all"), required=True
    )
    parser.add_argument("--encoder-checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    cfg = _apply_overrides(_load_config(args.config), args.overrides)
    seed = int(cfg.get("seed", 0))
    rank, world_size, device = setup_distributed()
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)
    strategy = str(cfg.get("tensor_sharing_strategy", "file_descriptor"))
    if int(cfg.get("num_workers", 0)) > 0:
        torch.multiprocessing.set_sharing_strategy(strategy)
    protected_roots = cfg.get("protected_data_roots", ())
    if args.stage == "all":
        if args.resume:
            parser.error("--resume is not allowed with --stage all")
        root = Path(args.output or cfg.get(
            "output_root",
            "data/runs/tactile_contextmask_best",
        ))
        encoder_output = ensure_writable_output(
            unique_output_dir(root / "stage_a"), protected_roots
        )
        predictor_output = ensure_writable_output(
            unique_output_dir(root / "stage_b"), protected_roots
        )
        if rank == 0:
            print(
                f"stage=all encoder_output={encoder_output} "
                f"predictor_output={predictor_output}",
                flush=True,
            )
        encoder_checkpoint = run_encoder_stage(
            cfg,
            device=device,
            rank=rank,
            world_size=world_size,
            output=encoder_output,
            resume=None,
            max_steps=args.max_steps,
            max_records=args.max_records,
        )
        checkpoint = run_predictor_stage(
            cfg,
            device=device,
            rank=rank,
            world_size=world_size,
            output=predictor_output,
            encoder_path=str(encoder_checkpoint),
            resume=None,
            max_steps=args.max_steps,
            max_records=args.max_records,
        )
    else:
        if args.stage == "encoder":
            output = args.output or cfg["encoder_training"]["output"]
            output = ensure_writable_output(
                unique_output_dir(output), protected_roots
            )
            if rank == 0:
                print(f"stage=encoder output={output}", flush=True)
            checkpoint = run_encoder_stage(
                cfg,
                device=device,
                rank=rank,
                world_size=world_size,
                output=output,
                resume=args.resume,
                max_steps=args.max_steps,
                max_records=args.max_records,
            )
        else:
            output = args.output or cfg["predictor_training"]["output"]
            output = ensure_writable_output(
                unique_output_dir(output), protected_roots
            )
            encoder_path = args.encoder_checkpoint
            if encoder_path is None:
                encoder_path = str(
                    Path(cfg["encoder_training"]["output"]) / "best_encoder.pt"
                )
            if rank == 0:
                print(f"stage=predictor output={output}", flush=True)
            checkpoint = run_predictor_stage(
                cfg,
                device=device,
                rank=rank,
                world_size=world_size,
                output=output,
                encoder_path=encoder_path,
                resume=args.resume,
                max_steps=args.max_steps,
                max_records=args.max_records,
            )
    if rank == 0:
        print(f"finished stage={args.stage} checkpoint={checkpoint}", flush=True)


if __name__ == "__main__":
    main()
