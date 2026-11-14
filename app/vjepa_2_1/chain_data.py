"""Datasets for independent tactile chains and four-chain episodes."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from app.vjepa_2_1.data import (
    RobotVisionTactileDataset,
    _as_float_tensor,
    _load_parts,
    _load_window,
    _path_parts,
    _sequence_length,
)


class ChainEpisodeDataset(Dataset):
    """Load one episode with up to four tactile chains.

    Manifest list order is fixed to:
    left_hand_left_sensor, left_hand_right_sensor,
    right_hand_left_sensor, right_hand_right_sensor.
    Missing entries are zero-padded and masked by ``chain_valid``.
    """

    def __init__(
        self,
        manifest,
        *,
        context=4,
        horizon=34,
        items=None,
        windows_per_episode=1,
        random_crop=True,
        require_contact=True,
        min_contact_frames=34,
        min_chain_contact_frames=1,
        load_vision=True,
        vision_size=384,
        normalize_vision=True,
        load_state=True,
        defer_image_conversion=False,
        balance_single_dual=True,
        dual_window_probability=0.5,
    ):
        self.root = Path(manifest).parent
        if items is None:
            items = [
                json.loads(line)
                for line in Path(manifest).open(encoding="utf-8")
                if line.strip()
            ]
        self.items = list(items)
        self.context = int(context)
        self.horizon = int(horizon)
        self.load_horizon = self.horizon
        self.windows_per_episode = max(1, int(windows_per_episode))
        self.random_crop = bool(random_crop)
        self.require_contact = bool(require_contact)
        self.min_contact_frames = max(1, int(min_contact_frames))
        self.min_chain_contact_frames = max(0, int(min_chain_contact_frames))
        self.load_vision = bool(load_vision)
        self.vision_size = int(vision_size)
        self.normalize_vision = bool(normalize_vision)
        self.load_state = bool(load_state)
        self.defer_image_conversion = bool(defer_image_conversion)
        self.balance_single_dual = bool(balance_single_dual)
        self.dual_window_probability = float(dual_window_probability)
        self._usable_length_cache = {}
        self._contact_window_cache = {}
        self._window_groups_cache = {}
        self.vision_helper = RobotVisionTactileDataset(
            manifest,
            context=self.context,
            horizon=self.horizon,
            items=self.items[:1],
            load_vision=False,
            load_state=False,
            defer_image_conversion=True,
        )

    def __len__(self):
        return len(self.items) * self.windows_per_episode

    def _resolve(self, value) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _available_paths(self, item: dict, key: str):
        values = item.get(key)
        if not isinstance(values, list):
            values = [values] * 4
        return [
            str(self._resolve(path)) if path else None
            for path in (list(values) + [None] * 4)[:4]
        ]

    def _usable_length(self, item: dict) -> int:
        cache_key = str(item["group"])
        cached = self._usable_length_cache.get(cache_key)
        if cached is not None:
            return int(cached)
        lengths = []
        if self.load_vision:
            for group in item.get("vision_paths", []):
                if group:
                    lengths.extend(
                        _sequence_length(self._resolve(path))
                        for path in _path_parts(group)
                    )
        for path in self._available_paths(item, "tactile"):
            if path:
                lengths.append(_sequence_length(Path(path)))
        for path in self._available_paths(item, "contact"):
            if path:
                lengths.append(_sequence_length(Path(path)))
        if self.load_state:
            for group in item.get("state_paths", []):
                if group:
                    lengths.extend(
                        _sequence_length(self._resolve(path))
                        for path in _path_parts(group)
                    )
        length = min(lengths)
        self._usable_length_cache[cache_key] = length
        return length

    def _contact_matrix(self, item: dict) -> np.ndarray:
        paths = self._available_paths(item, "contact")
        present = self._available_paths(item, "tactile")
        loaded = []
        for path in paths:
            if path is None:
                loaded.append(None)
            else:
                loaded.append(
                    np.asarray(
                        np.load(path, mmap_mode="r", allow_pickle=False)
                    ).reshape(-1)
                    > 0
                )
        lengths = [len(value) for value in loaded if value is not None]
        if not lengths:
            if self.require_contact:
                raise ValueError(f"episode {item['group']!r} has no contact labels")
            return np.zeros((0, 4), dtype=bool)
        length = min(lengths)
        matrix = np.zeros((length, 4), dtype=bool)
        for chain, value in enumerate(loaded):
            if present[chain] and value is not None:
                matrix[:, chain] = value[:length]
        return matrix

    def contact_window_starts(self, item: dict) -> np.ndarray:
        cache_key = str(item["group"])
        cached = self._contact_window_cache.get(cache_key)
        if cached is not None:
            return cached
        contact = self._contact_matrix(item)
        n = self.context + self.horizon
        if len(contact) < n:
            starts = np.zeros(0, dtype=np.int64)
            self._contact_window_cache[cache_key] = starts
            return starts
        windows = np.lib.stride_tricks.sliding_window_view(
            contact.astype(np.int32), n, axis=0
        )
        chain_frames = windows.sum(axis=-1)
        valid_chains = chain_frames >= self.min_chain_contact_frames
        left_active = valid_chains[:, :2].any(axis=-1)
        right_active = valid_chains[:, 2:].any(axis=-1)
        any_contact = (windows > 0).any(axis=1)
        contact_frames = any_contact.sum(axis=-1)
        keep = valid_chains.any(axis=-1) & (
            contact_frames >= self.min_contact_frames
        )
        starts = np.flatnonzero(keep)
        self._contact_window_cache[cache_key] = starts
        self._window_groups_cache[cache_key] = {
            "dual": starts[left_active[starts] & right_active[starts]],
            "single": starts[left_active[starts] ^ right_active[starts]],
        }
        return starts

    def __getitem__(self, index):
        item_index = int(index) // self.windows_per_episode
        local_window = int(index) % self.windows_per_episode
        item = self.items[item_index]
        n = self.context + self.horizon
        usable = self._usable_length(item)
        if usable < n:
            raise ValueError(
                f"episode {item['group']!r} has {usable} frames < {n}"
            )
        starts = self.contact_window_starts(item) if self.require_contact else None
        if starts is not None:
            if not len(starts):
                raise ValueError(f"episode {item['group']!r} has no contact window")
            groups = self._window_groups_cache.get(str(item["group"]), {})
            dual = groups.get("dual", np.zeros(0, dtype=np.int64))
            single = groups.get("single", np.zeros(0, dtype=np.int64))
            if self.random_crop:
                candidates = starts
                if (
                    self.balance_single_dual
                    and len(dual)
                    and random.random() < self.dual_window_probability
                ):
                    candidates = dual
                elif self.balance_single_dual and len(single):
                    candidates = single
                start = int(random.choice(candidates.tolist()))
            else:
                candidates = starts
                if self.balance_single_dual:
                    if local_window % 2 == 0 and len(dual):
                        candidates = dual
                    elif len(single):
                        candidates = single
                start = int(
                    candidates[(local_window + index) % len(candidates)]
                )
        elif self.random_crop:
            start = random.randint(0, usable - n)
        else:
            start = 0

        tactile_paths = self._available_paths(item, "tactile")
        contact_paths = self._available_paths(item, "contact")
        slip_paths = self._available_paths(item, "slip")
        tactile = []
        template = None
        for path in tactile_paths:
            if path is None:
                continue
            value = _load_window(Path(path), start, n)
            if not self.defer_image_conversion:
                value = _as_float_tensor(value)
            value = self.vision_helper._prepare_tactile(value)
            template = value
            tactile.append(value)
            break
        if template is None:
            raise ValueError(f"episode {item['group']!r} has no tactile path")
        padded = []
        for path in tactile_paths:
            if path is None:
                padded.append(torch.zeros_like(template))
            else:
                value = _load_window(Path(path), start, n)
                if not self.defer_image_conversion:
                    value = _as_float_tensor(value)
                padded.append(self.vision_helper._prepare_tactile(value))
        tactile = torch.stack(padded, dim=0)

        contact = np.zeros((n, 4), dtype=np.float32)
        slip = np.zeros((n, 4), dtype=np.float32)
        for chain, (cpath, spath) in enumerate(zip(contact_paths, slip_paths)):
            if cpath is None or spath is None:
                continue
            cvalue = np.asarray(_load_window(Path(cpath), start, n)).reshape(n)
            svalue = np.asarray(_load_window(Path(spath), start, n)).reshape(n)
            contact[:, chain] = cvalue > 0
            slip[:, chain] = svalue > 0
        chain_valid = (
            torch.tensor([path is not None for path in tactile_paths])
            & torch.tensor(contact.sum(axis=0) >= self.min_chain_contact_frames)
        )

        out = {
            "tactile_chains": tactile,
            "chain_valid": chain_valid,
            "contact": torch.from_numpy(contact),
            "slip": torch.from_numpy(slip),
            "group": item["group"],
        }
        if self.load_vision:
            vision_arms = []
            template_vision = None
            for group in item.get("vision_paths", [None, None]):
                if group:
                    vision = _load_parts(
                        tuple(
                            self._resolve(path)
                            for path in _path_parts(group)
                        ),
                        start,
                        n,
                        concat_dim=-1,
                    )
                    if not self.defer_image_conversion:
                        vision = _as_float_tensor(vision)
                        vision = self.vision_helper._prepare_vision(vision)
                    template_vision = vision
                else:
                    vision = (
                        torch.zeros_like(template_vision)
                        if template_vision is not None
                        else None
                    )
                vision_arms.append(vision)
            if vision_arms[0] is None:
                vision_arms[0] = vision_arms[1]
            if vision_arms[1] is None:
                vision_arms[1] = vision_arms[0]
            out["vision_arms"] = torch.stack(vision_arms, dim=0)
            out["arm_valid"] = torch.tensor(item.get("arm_valid", [True, True]))
            out["vision"] = out["vision_arms"][0]
        if self.load_state:
            state_arms = []
            template_state = None
            for group in item.get("state_paths", [None, None]):
                if group:
                    state = _load_parts(
                        tuple(
                            self._resolve(path)
                            for path in _path_parts(group)
                        ),
                        start,
                        n,
                        concat_dim=-1,
                    ).float()
                    template_state = state
                else:
                    state = (
                        torch.zeros_like(template_state)
                        if template_state is not None
                        else None
                    )
                state_arms.append(state)
            if state_arms[0] is None:
                state_arms[0] = state_arms[1]
            if state_arms[1] is None:
                state_arms[1] = state_arms[0]
            out["state_arms"] = torch.stack(state_arms, dim=0)
            out["arm_valid"] = torch.tensor(item.get("arm_valid", [True, True]))
            out["state"] = out["state_arms"][0]
        return out
