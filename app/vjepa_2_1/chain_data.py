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
        window_overlap=1.0 / 3.0,
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
        window_sampling="contact",
        no_contact_fraction=None,
        length_source="all",
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
        windows_per_episode = int(windows_per_episode)
        if windows_per_episode < 0:
            raise ValueError("windows_per_episode must be non-negative")
        self.windows_per_episode = windows_per_episode
        self.dense_windows = windows_per_episode == 0
        self.window_overlap = float(window_overlap)
        if not 0.0 <= self.window_overlap < 1.0:
            raise ValueError("window_overlap must be in [0, 1)")
        self.window_sampling = str(window_sampling).lower()
        if self.window_sampling not in ("contact", "all", "mixed", "stratified"):
            raise ValueError(
                "window_sampling must be contact, all, mixed, or stratified"
            )
        if no_contact_fraction is not None:
            no_contact_fraction = float(no_contact_fraction)
            if not 0.0 <= no_contact_fraction < 1.0:
                raise ValueError(
                    "no_contact_fraction must be in [0, 1)"
                )
        self.no_contact_fraction = no_contact_fraction
        self.length_source = str(length_source).lower()
        if self.length_source not in ("all", "contact"):
            raise ValueError("length_source must be all or contact")
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
        self._window_category_cache = {}
        self._all_start_cache = {}
        self._dense_start_cache = {}
        self._dense_window_counts = None
        self._path_exists_cache = {}
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
        if self.dense_windows:
            return int(self._dense_counts().sum())
        return len(self.items) * self.windows_per_episode

    def _resolve(self, value) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _available_paths(self, item: dict, key: str):
        values = item.get(key)
        if not isinstance(values, list):
            values = [values] * 4
        paths = []
        for path in (list(values) + [None] * 4)[:4]:
            if not path:
                paths.append(None)
                continue
            resolved = str(self._resolve(path))
            exists = self._path_exists_cache.get(resolved)
            if exists is None:
                exists = Path(resolved).is_file()
                self._path_exists_cache[resolved] = exists
            paths.append(resolved if exists else None)
        return paths

    def _available_path_group(self, group) -> bool:
        return bool(group) and all(
            self._path_exists(Path(path))
            for path in _path_parts(group)
        )

    def _path_exists(self, path: Path) -> bool:
        resolved = str(self._resolve(path))
        exists = self._path_exists_cache.get(resolved)
        if exists is None:
            exists = Path(resolved).is_file()
            self._path_exists_cache[resolved] = exists
        return exists

    def _usable_length(self, item: dict) -> int:
        cache_key = str(item["group"])
        cached = self._usable_length_cache.get(cache_key)
        if cached is not None:
            return int(cached)
        if self.length_source == "contact":
            contact_lengths = [
                _sequence_length(Path(path))
                for path in self._available_paths(item, "contact")
                if path
            ]
            if contact_lengths:
                length = min(contact_lengths)
                self._usable_length_cache[cache_key] = length
                return length
        lengths = []
        if self.load_vision:
            for group in item.get("vision_paths", []):
                if self._available_path_group(group):
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
                if self._available_path_group(group):
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

    def _all_window_starts(self, item: dict) -> np.ndarray:
        cache_key = str(item["group"])
        cached = self._all_start_cache.get(cache_key)
        if cached is not None:
            return cached
        count = self.context + self.horizon
        usable = self._usable_length(item)
        starts = (
            self._dense_starts(
                np.arange(
                    0,
                    usable - count + 1,
                    dtype=np.int64,
                ),
                count,
            )
            if usable >= count
            else np.zeros(0, dtype=np.int64)
        )
        self._all_start_cache[cache_key] = starts
        return starts

    def _categorized_window_starts(
        self, item: dict
    ) -> dict[str, np.ndarray]:
        cache_key = str(item["group"])
        cached = self._window_category_cache.get(cache_key)
        if cached is not None:
            return cached
        contact = self._contact_matrix(item)
        count = self.context + self.horizon
        if len(contact) < count:
            categories = {
                name: np.zeros(0, dtype=np.int64)
                for name in (
                    "no_contact",
                    "start",
                    "ongoing",
                    "end",
                )
            }
            self._window_category_cache[cache_key] = categories
            return categories
        all_starts = np.arange(
            0, len(contact) - count + 1, dtype=np.int64
        )
        windows = np.lib.stride_tricks.sliding_window_view(
            contact.astype(np.int32), count, axis=0
        )[all_starts].transpose(0, 2, 1)
        history_contact = windows[:, : self.context].any(axis=(1, 2))
        future_contact = windows[:, self.context :].any(axis=(1, 2))
        categories = {
            "no_contact": all_starts[
                ~history_contact & ~future_contact
            ],
            "start": all_starts[~history_contact & future_contact],
            "ongoing": all_starts[history_contact & future_contact],
            "end": all_starts[history_contact & ~future_contact],
        }
        self._window_category_cache[cache_key] = categories
        return categories

    @staticmethod
    def _uniform_subset(
        values: np.ndarray, target: int
    ) -> np.ndarray:
        values = np.asarray(values, dtype=np.int64)
        target = int(target)
        if target <= 0 or len(values) == 0:
            return values[:0]
        if target >= len(values):
            return values
        positions = np.rint(
            np.linspace(0, len(values) - 1, target)
        ).astype(np.int64)
        return values[np.unique(positions)]

    def _item_dense_starts(self, item: dict) -> np.ndarray:
        cache_key = str(item["group"])
        cached = self._dense_start_cache.get(cache_key)
        if cached is not None:
            return cached
        count = self.context + self.horizon
        if self.window_sampling == "contact":
            candidates = self.contact_window_starts(item)
            starts = self._dense_starts(candidates, count)
        elif self.window_sampling == "all":
            starts = self._all_window_starts(item)
        elif self.window_sampling == "mixed":
            all_starts = self._all_window_starts(item)
            if (
                self.no_contact_fraction is None
                or len(all_starts) == 0
            ):
                starts = all_starts
            else:
                categories = self._categorized_window_starts(item)
                no_contact = self._dense_starts(
                    categories["no_contact"], count
                )
                contact = np.sort(
                    np.concatenate(
                        (
                            categories["start"],
                            categories["ongoing"],
                            categories["end"],
                        )
                    )
                )
                contact = self._dense_starts(contact, count)
                if len(contact) == 0:
                    starts = no_contact
                else:
                    fraction = self.no_contact_fraction
                    target_no_contact = int(
                        len(contact)
                        * fraction
                        / max(1e-6, 1.0 - fraction)
                    )
                    no_contact = self._uniform_subset(
                        no_contact, target_no_contact
                    )
                    starts = np.sort(
                        np.unique(
                            np.concatenate((contact, no_contact))
                        )
                    )
        else:
            raise ValueError(
                "window_sampling='stratified' is only valid for fixed "
                "validation/test window counts"
            )
        self._dense_start_cache[cache_key] = starts
        return starts

    def _dense_counts(self) -> np.ndarray:
        if self._dense_window_counts is None:
            self._dense_window_counts = np.asarray(
                [len(self._item_dense_starts(item)) for item in self.items],
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

    def __getitem__(self, index):
        if self.dense_windows:
            item_index, local_window = self._locate_dense_index(int(index))
        else:
            item_index = int(index) // self.windows_per_episode
            local_window = int(index) % self.windows_per_episode
        item = self.items[item_index]
        n = self.context + self.horizon
        usable = self._usable_length(item)
        if usable < n:
            raise ValueError(
                f"episode {item['group']!r} has {usable} frames < {n}"
            )
        if self.dense_windows:
            dense_starts = self._item_dense_starts(item)
            if len(dense_starts) == 0:
                raise ValueError(
                    f"episode {item['group']!r} has no usable window"
                )
            start = int(dense_starts[local_window])
        elif self.random_crop:
            if self.window_sampling == "contact":
                starts = self.contact_window_starts(item)
                if not len(starts):
                    raise ValueError(
                        f"episode {item['group']!r} has no contact window"
                    )
                start = int(random.choice(starts.tolist()))
            else:
                start = random.randint(0, usable - n)
        elif self.window_sampling == "stratified":
            categories = self._categorized_window_starts(item)
            category_order = ("ongoing", "no_contact", "start", "end")
            candidates = np.zeros(0, dtype=np.int64)
            for offset in range(len(category_order)):
                category = category_order[
                    (item_index + local_window + offset)
                    % len(category_order)
                ]
                candidates = categories[category]
                if len(candidates):
                    break
            if len(candidates) == 0:
                raise ValueError(
                    f"episode {item['group']!r} has no usable window"
                )
            start = int(
                candidates[(local_window + index) % len(candidates)]
            )
        elif self.window_sampling == "contact":
            starts = self.contact_window_starts(item)
            if not len(starts):
                raise ValueError(
                    f"episode {item['group']!r} has no contact window"
                )
            groups = self._window_groups_cache.get(str(item["group"]), {})
            dual = groups.get("dual", np.zeros(0, dtype=np.int64))
            single = groups.get("single", np.zeros(0, dtype=np.int64))
            candidates = starts
            if self.balance_single_dual:
                if local_window % 2 == 0 and len(dual):
                    candidates = dual
                elif len(single):
                    candidates = single
            start = int(
                candidates[(local_window + index) % len(candidates)]
            )
        else:
            starts = self._all_window_starts(item)
            if len(starts) == 0:
                raise ValueError(
                    f"episode {item['group']!r} has no usable window"
                )
            start = int(starts[(local_window + index) % len(starts)])

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
        manifest_chain_valid = item.get(
            "chain_valid", [True] * len(tactile_paths)
        )
        chain_valid = torch.tensor(
            [path is not None for path in tactile_paths]
        ) & torch.tensor(
            [bool(value) for value in manifest_chain_valid[:4]]
        )

        out = {
            "tactile_chains": tactile,
            "chain_valid": chain_valid,
            "contact": torch.from_numpy(contact),
            "slip": torch.from_numpy(slip),
            "group": item["group"],
        }
        vision_groups = list(item.get("vision_paths", [None, None]))[:2]
        state_groups = list(item.get("state_paths", [None, None]))[:2]
        vision_available = torch.tensor(
            [self._available_path_group(group) for group in vision_groups]
            + [False] * (2 - len(vision_groups))
        )
        state_available = torch.tensor(
            [self._available_path_group(group) for group in state_groups]
            + [False] * (2 - len(state_groups))
        )
        manifest_arm_valid = torch.tensor(
            item.get("arm_valid", [True, True]), dtype=torch.bool
        )
        arm_valid = manifest_arm_valid & (
            vision_available | state_available
        )
        if self.load_vision:
            vision_arms = []
            template_vision = None
            for group in item.get("vision_paths", [None, None]):
                if self._available_path_group(group):
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
            out["vision"] = out["vision_arms"][0]
        if self.load_state:
            state_arms = []
            template_state = None
            for group in item.get("state_paths", [None, None]):
                if self._available_path_group(group):
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
            out["state"] = out["state_arms"][0]
        out["arm_valid"] = arm_valid
        return out
