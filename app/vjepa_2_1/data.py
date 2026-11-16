"""基于 JSONL manifest 的同步机器人视触数据集。

每条记录可以指向 ``.npy``、``.pt`` 或 ``.mp4``。视觉和触觉按同一时间窗
懒加载；多路触觉/状态会沿数据轴拼接。这样不会像旧实现那样在每个
``__getitem__`` 中把整段 episode 载入内存。

支持的记录格式::

    {"vision": "...mp4", "tactile": "...mp4", "state": "...npy", "group": "task/episode"}
    {"vision": "...mp4",
     "tactile": [".../tactile_left.mp4", ".../tactile_right.mp4"],
     "state": [".../vio_pose.npy", ".../gripper_pose.npy"],
     "group": "task/episode"}
"""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _path_parts(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, Path)):
        return (str(value),)
    if isinstance(value, Sequence):
        parts = tuple(str(x) for x in value)
        if not parts:
            raise ValueError("path list cannot be empty")
        return parts
    raise TypeError(f"expected a path or list of paths, got {type(value).__name__}")


def _infer_tactile_side(path: str | Path) -> str | None:
    name = Path(path).name.lower()
    if "right" in name:
        return "right"
    if "left" in name:
        return "left"
    return None


def _tactile_sides(record: dict) -> tuple[str, ...]:
    paths = _path_parts(record["tactile"])
    configured = record.get("tactile_sides")
    if configured is not None:
        sides = tuple(str(side).lower() for side in configured)
        if len(sides) != len(paths) or any(side not in ("left", "right") for side in sides):
            raise ValueError(
                f"invalid tactile_sides for record {record.get('vision')!r}: {sides}"
            )
        return sides
    inferred = tuple(_infer_tactile_side(path) for path in paths)
    if len(paths) == 2 and all(side is None for side in inferred):
        return ("left", "right")
    if len(paths) == 1 and inferred[0] is None:
        return ("left",)
    if all(side is not None for side in inferred):
        return inferred
    raise ValueError(
        f"cannot infer tactile side for {record.get('vision')!r}; "
        "add tactile_sides=[left,right] to the manifest"
    )


@lru_cache(maxsize=32768)
def _video_length(path: str) -> int:
    try:
        import decord

        return len(decord.VideoReader(path, ctx=decord.cpu(0), num_threads=1))
    except ImportError:
        import cv2

        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise OSError(f"could not open video {path}")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if count <= 0:
            raise OSError(f"could not determine video length for {path}")
        return count


def _sequence_length(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix in _VIDEO_SUFFIXES:
        return _video_length(str(path.resolve()))
    if suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return int(len(array))
    if suffix in (".pt", ".pth"):
        return int(len(torch.load(path, map_location="cpu", weights_only=False)))
    raise ValueError(f"unsupported frame source {path}; expected npy/pt/video")


def _decode_video_window(path: Path, start: int, count: int) -> np.ndarray:
    """Decode a contiguous THWC window without loading the complete video."""
    try:
        import decord

        reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
        return reader.get_batch(np.arange(start, start + count, dtype=np.int64)).asnumpy()
    except ImportError:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise OSError(f"could not open video {path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(count):
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()
        if len(frames) != count:
            raise OSError(f"could only decode {len(frames)}/{count} frames from {path}")
        return np.stack(frames, axis=0)


def _load_window(path: Path, start: int, count: int) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix in _VIDEO_SUFFIXES:
        frames = _decode_video_window(path, start, count)
        # THWC -> TCHW.
        return torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
    if suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        # np.array materializes only the requested window.
        window = np.array(array[start:start + count], copy=True)
        return torch.from_numpy(window)
    if suffix in (".pt", ".pth"):
        array = torch.load(path, map_location="cpu", weights_only=False)
        return torch.as_tensor(array[start:start + count])
    raise ValueError(f"unsupported frame source {path}; expected npy/pt/video")


def _load_parts(paths: Sequence[Path], start: int, count: int,
                *, concat_dim: int) -> torch.Tensor:
    parts = [_load_window(path, start, count) for path in paths]
    if len(parts) == 1:
        return parts[0]
    max_ndim = max(part.ndim for part in parts)
    aligned = [
        part[:, None] if part.ndim == 1 and max_ndim == 2 else part
        for part in parts
    ]
    return torch.cat(aligned, dim=concat_dim)


def _as_float_tensor(raw: torch.Tensor) -> torch.Tensor:
    """Convert image-like values to float without changing scalar state values."""
    x = raw.float()
    if x.ndim >= 3 and torch.isfinite(x).all() and x.min() >= 0 and x.max() > 1.5:
        x = x / 255.0
    return x


class RobotVisionTactileDataset(Dataset):
    """Lazily read synchronized visual/tactile windows from a manifest.

    ``windows_per_episode`` controls how many windows one manifest record
    contributes per epoch. Training windows are random; validation windows are
    evenly spaced across the episode. This prevents a long dataset from exposing
    only one fixed validation crop per episode.
    """

    def __init__(
        self,
        manifest,
        context=8,
        horizon=4,
        transform=None,
        tactile_transform=None,
        state_transform=None,
        random_crop=True,
        require_state=False,
        vision_size=None,
        normalize_vision=True,
        vision_mean=(0.485, 0.456, 0.406),
        vision_std=(0.229, 0.224, 0.225),
        items=None,
        windows_per_episode=1,
        window_overlap=1.0 / 3.0,
        load_horizon=None,
        load_vision=True,
        load_vision_frames=None,
        load_state=True,
        defer_image_conversion=False,
        require_contact=False,
        min_contact_frames=2,
        num_hands=None,
        min_hand_contact_frames=1,
    ):
        self.root = Path(manifest).parent
        if items is None:
            with open(manifest, "r", encoding="utf-8") as f:
                self.items = [json.loads(x) for x in f if x.strip()]
        else:
            self.items = list(items)
        if not self.items:
            raise ValueError("manifest is empty")
        if any("vision" not in r or "tactile" not in r for r in self.items):
            raise ValueError("each manifest record must contain vision and tactile paths")
        if require_state and any("state" not in r for r in self.items):
            raise ValueError("state is required: add state paths to every manifest record")
        if context <= 0 or horizon <= 0:
            raise ValueError("context and horizon must be positive")
        load_horizon = horizon if load_horizon is None else int(load_horizon)
        if load_horizon < 0:
            raise ValueError("load_horizon must be non-negative")
        windows_per_episode = int(windows_per_episode)
        if windows_per_episode < 0:
            raise ValueError("windows_per_episode must be non-negative")

        self.context, self.horizon = int(context), int(horizon)
        self.load_horizon = load_horizon
        self.transform, self.tactile_transform = transform, tactile_transform
        self.state_transform, self.require_state = state_transform, require_state
        self.random_crop = bool(random_crop)
        self.vision_size = vision_size
        self.normalize_vision = bool(normalize_vision)
        self.load_vision = bool(load_vision)
        self.load_vision_frames = (
            None if load_vision_frames is None else max(1, int(load_vision_frames))
        )
        self.load_state = bool(load_state)
        self.defer_image_conversion = bool(defer_image_conversion)
        self.require_contact = bool(require_contact)
        self.min_contact_frames = max(1, int(min_contact_frames))
        self.num_hands = None if num_hands is None else int(num_hands)
        if self.num_hands not in (None, 1, 2):
            raise ValueError("num_hands must be None, 1, or 2")
        self.min_hand_contact_frames = max(0, int(min_hand_contact_frames))
        self.windows_per_episode = windows_per_episode
        self.dense_windows = windows_per_episode == 0
        self.window_overlap = float(window_overlap)
        if not 0.0 <= self.window_overlap < 1.0:
            raise ValueError("window_overlap must be in [0, 1)")
        self._usable_length_cache = {}
        self._contact_cache = {}
        self._dense_start_cache = {}
        self._dense_window_counts = None
        self.vision_mean = torch.tensor(vision_mean, dtype=torch.float32).view(1, -1, 1, 1)
        self.vision_std = torch.tensor(vision_std, dtype=torch.float32).view(1, -1, 1, 1)

    def __len__(self):
        if self.dense_windows:
            return int(self._dense_counts().sum())
        return len(self.items) * self.windows_per_episode

    def _resolve(self, path: str) -> Path:
        path = Path(path)
        return path if path.is_absolute() else self.root / path

    def _record_key(self, record: dict) -> tuple:
        contact = record.get("contact")
        return (
            str(record["vision"]),
            tuple(_path_parts(record["tactile"])),
            tuple(_path_parts(contact)) if contact else (),
        )

    def _sources(self, record: dict, key: str) -> tuple[Path, ...]:
        return tuple(self._resolve(path) for path in _path_parts(record[key]))

    def _usable_length(self, record: dict) -> int:
        cache_key = self._record_key(record)
        cached = self._usable_length_cache.get(cache_key)
        if cached is not None:
            return int(cached)
        lengths = [_sequence_length(path) for path in self._sources(record, "vision")]
        lengths += [_sequence_length(path) for path in self._sources(record, "tactile")]
        for key in ("state", "action", "contact", "slip"):
            if key in record:
                lengths += [_sequence_length(path) for path in self._sources(record, key)]
        length = min(lengths)
        self._usable_length_cache[cache_key] = length
        return length

    def _hand_presence(self, record: dict) -> np.ndarray:
        present = np.zeros(2, dtype=bool)
        for side in _tactile_sides(record):
            present[0 if side == "left" else 1] = True
        return present

    def _label_matrix(self, record: dict, key: str, start: int, count: int) -> np.ndarray | None:
        sources = record.get(key)
        if not sources:
            return None
        paths = _path_parts(sources)
        sides = _tactile_sides(record)
        if len(paths) != len(sides):
            raise ValueError(
                f"{key!r} sources for {record.get('vision')!r} must align with tactile streams"
            )
        matrix = np.zeros((count, 2), dtype=np.float32)
        for path, side in zip(paths, sides):
            values = np.asarray(
                _load_window(self._resolve(path), start, count)
            ).reshape(count)
            matrix[:, 0 if side == "left" else 1] = values > 0
        return matrix

    def _contact_matrix(self, record: dict) -> np.ndarray:
        sources = record.get("contact")
        if not sources:
            if self.require_contact:
                raise ValueError(
                    f"record {record.get('vision')!r} has no contact mask"
                )
            return np.zeros((0, 2), dtype=np.float32)
        length = min(
            _sequence_length(self._resolve(path))
            for path in _path_parts(sources)
        )
        return self._label_matrix(record, "contact", 0, length)

    def contact_window_starts(self, record: dict) -> np.ndarray:
        """Return window starts satisfying the configured contact minimum."""
        cache_key = self._record_key(record)
        cached = self._contact_cache.get(cache_key)
        if cached is not None:
            return cached
        if not self.require_contact:
            raise ValueError("contact_window_starts requires require_contact=True")
        contact = self._contact_matrix(record)
        n = self.context + self.load_horizon
        if contact.shape[0] < n:
            return np.zeros(0, dtype=np.int64)
        windows = np.lib.stride_tricks.sliding_window_view(
            contact.astype(np.int32), n, axis=0
        )
        # [starts, hands, time]
        hand_counts = windows.sum(axis=-1)
        valid_hands = self._hand_presence(record)[None] & (
            hand_counts >= self.min_hand_contact_frames
        )
        any_contact = (windows > 0).any(axis=1)
        contact_frames = any_contact.sum(axis=-1)
        keep = valid_hands.any(axis=-1) & (
            contact_frames >= self.min_contact_frames
        )
        starts = np.flatnonzero(keep)
        self._contact_cache[cache_key] = starts
        return starts

    def _window_stride(self, count: int) -> int:
        return max(
            1,
            int(round(int(count) * (1.0 - self.window_overlap))),
        )

    def _dense_starts(
        self, candidates: np.ndarray, count: int
    ) -> np.ndarray:
        candidates = np.asarray(candidates, dtype=np.int64)
        if len(candidates) == 0:
            return candidates
        stride = self._window_stride(count)
        selected = []
        next_start = int(candidates[0])
        for start in candidates:
            start = int(start)
            if start >= next_start:
                selected.append(start)
                next_start = start + stride
        if selected[-1] != int(candidates[-1]):
            selected.append(int(candidates[-1]))
        return np.asarray(selected, dtype=np.int64)

    def _item_dense_starts(self, record: dict) -> np.ndarray:
        cache_key = self._record_key(record)
        cached = self._dense_start_cache.get(cache_key)
        if cached is not None:
            return cached
        count = self.context + self.load_horizon
        if self.require_contact:
            candidates = self.contact_window_starts(record)
        else:
            usable = self._usable_length(record)
            candidates = (
                np.arange(0, usable - count + 1, dtype=np.int64)
                if usable >= count
                else np.zeros(0, dtype=np.int64)
            )
        starts = self._dense_starts(candidates, count)
        self._dense_start_cache[cache_key] = starts
        return starts

    def _dense_counts(self) -> np.ndarray:
        if self._dense_window_counts is None:
            self._dense_window_counts = np.asarray(
                [len(self._item_dense_starts(record)) for record in self.items],
                dtype=np.int64,
            )
        return self._dense_window_counts

    def _locate_dense_index(self, index: int) -> tuple[int, int]:
        counts = self._dense_counts()
        total = int(counts.sum())
        if index < 0:
            index += total
        if index < 0 or index >= total:
            raise IndexError(index)
        cumulative = np.cumsum(counts)
        item_index = int(np.searchsorted(cumulative, index, side="right"))
        previous = int(cumulative[item_index - 1]) if item_index else 0
        return item_index, index - previous

    def _prepare_vision(self, vision: torch.Tensor) -> torch.Tensor:
        """返回符合 V-JEPA 约定的 ``[T,C,H,W]`` 浮点输入。"""
        if vision.ndim != 4:
            raise ValueError(f"vision must be a 4-D frame sequence, got {tuple(vision.shape)}")
        # 新 manifest 约定是 TCHW。同时接受常见的 THWC / CTHW。
        if vision.shape[-1] in (1, 3) and vision.shape[1] > 3:
            vision = vision.permute(0, 3, 1, 2)
        elif vision.shape[0] in (1, 3) and vision.shape[1] > 3:
            vision = vision.permute(1, 0, 2, 3)
        if vision.shape[1] not in (1, 3):
            raise ValueError(f"vision must have 1 or 3 channels, got shape {tuple(vision.shape)}")
        if vision.shape[1] == 1:
            vision = vision.repeat(1, 3, 1, 1)
        if self.vision_size is not None:
            size = ((int(self.vision_size),) * 2 if isinstance(self.vision_size, int)
                    else tuple(self.vision_size))
            if tuple(vision.shape[-2:]) != size:
                vision = F.interpolate(vision, size=size, mode="bilinear", align_corners=False)
        if self.normalize_vision:
            # 外部 transform 可能已经归一化过，避免再套一层 ImageNet mean/std。
            if float(vision.min()) >= -1e-3 and float(vision.max()) <= 1.001:
                mean = self.vision_mean.to(device=vision.device, dtype=vision.dtype)
                std = self.vision_std.to(device=vision.device, dtype=vision.dtype)
                vision = (vision - mean) / std
        return vision

    @staticmethod
    def _prepare_tactile(tactile: torch.Tensor) -> torch.Tensor:
        """触觉图像序列保持 ``[T,C,H,W]``。"""
        if tactile.ndim == 4 and tactile.shape[0] in (1, 2, 3) and tactile.shape[1] > 3:
            tactile = tactile.permute(1, 0, 2, 3)
        if tactile.ndim not in (3, 4):
            raise ValueError(f"tactile must be [T,C,H,W] or [T,F], got {tuple(tactile.shape)}")
        return tactile

    def __getitem__(self, index):
        if self.dense_windows:
            item_index, local_window = self._locate_dense_index(int(index))
        else:
            item_index = int(index) // self.windows_per_episode
            local_window = int(index) % self.windows_per_episode
        record = self.items[item_index]
        n = self.context + self.load_horizon
        usable = self._usable_length(record)
        if usable < n:
            raise ValueError(
                f"sequence {record.get('group', record['vision'])!r} has {usable} frames, "
                f"shorter than context+horizon={n}"
            )
        contact_starts = None
        if self.require_contact:
            contact_starts = self.contact_window_starts(record)
            if len(contact_starts) == 0:
                raise ValueError(
                    f"record {record.get('vision')!r} has no contact-rich window"
                )
        if self.dense_windows:
            dense_starts = self._item_dense_starts(record)
            if len(dense_starts) == 0:
                raise ValueError(
                    f"record {record.get('vision')!r} has no usable window"
                )
            start = int(dense_starts[local_window])
        elif contact_starts is not None:
            if self.random_crop:
                start = int(random.choice(contact_starts.tolist()))
            else:
                position = (local_window + index) % len(contact_starts)
                start = int(contact_starts[position])
        elif self.random_crop:
            start = random.randint(0, usable - n)
        elif self.windows_per_episode == 1:
            start = 0
        else:
            start = round(local_window * (usable - n) / (self.windows_per_episode - 1))
        start = int(min(max(0, start), usable - n))
        vision_count = (
            n if self.load_vision_frames is None
            else min(n, self.load_vision_frames)
        )
        vision = (
            _load_parts(
                self._sources(record, "vision"),
                start,
                vision_count,
                concat_dim=-1,
            )
            if self.load_vision else None
        )
        tactile_parts = [
            _load_window(path, start, n)
            for path in self._sources(record, "tactile")
        ]
        state = (_load_parts(self._sources(record, "state"), start, n, concat_dim=-1)
                 if self.load_state and "state" in record else None)
        action = (_load_parts(self._sources(record, "action"), start, n, concat_dim=-1)
                  if "action" in record else None)
        labels = {}
        for key in ("contact", "slip"):
            if key not in record:
                continue
            matrix = self._label_matrix(record, key, start, n)
            if self.num_hands != 2 and matrix.shape[1] == 2:
                active = matrix[:, self._hand_presence(record)]
                matrix = active[:, 0] if active.shape[1] == 1 else active
            labels[key] = torch.from_numpy(matrix)

        if vision is not None and not self.defer_image_conversion:
            vision = _as_float_tensor(vision)
        if not self.defer_image_conversion:
            tactile_parts = [_as_float_tensor(part) for part in tactile_parts]
        if state is not None:
            state = state.float()
            if state.ndim == 1:
                state = state[:, None]
        if action is not None:
            action = action.float()
        labels = {key: value.float() for key, value in labels.items()}

        if self.transform and vision is not None and not self.defer_image_conversion:
            vision = self.transform(vision)
        if vision is not None and not self.defer_image_conversion:
            vision = self._prepare_vision(vision)
        if self.tactile_transform:
            tactile_parts = [self.tactile_transform(part) for part in tactile_parts]
        tactile_parts = [self._prepare_tactile(part) for part in tactile_parts]
        sides = _tactile_sides(record)
        if len(tactile_parts) != len(sides):
            raise ValueError("tactile streams and tactile sides drifted out of sync")
        if self.num_hands == 2:
            prototype = tactile_parts[0]
            left = torch.zeros_like(prototype)
            right = torch.zeros_like(prototype)
            for part, side in zip(tactile_parts, sides):
                if side == "left":
                    left = part
                else:
                    right = part
            tactile_parts = [left, right]
            tactile = torch.cat((left, right), dim=-1)
        else:
            tactile = (
                tactile_parts[0] if len(tactile_parts) == 1
                else torch.cat(tactile_parts, dim=-1)
            )
        if self.state_transform and state is not None:
            state = self.state_transform(state)

        # 保持 [T,C,H,W]，训练循环再转成 V-JEPA 的 [B,C,T,H,W]。
        out = {"tactile": tactile, "context": self.context}
        if vision is not None:
            out["vision"] = vision
        if self.num_hands == 2:
            out["tactile_left"] = tactile_parts[0]
            out["tactile_right"] = tactile_parts[1]
            present = torch.from_numpy(self._hand_presence(record))
            if "contact" in labels:
                contact_counts = labels["contact"].sum(dim=0)
                present &= contact_counts >= self.min_hand_contact_frames
            out["hand_valid"] = present.bool()
        elif len(tactile_parts) == 2:
            out["tactile_left"] = tactile_parts[0]
            out["tactile_right"] = tactile_parts[1]
        if state is not None:
            out["state"] = state
        if action is not None:
            out["action"] = action
        out.update(labels)
        return out
