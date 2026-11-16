"""Episode-grouped train/validation/test splitting."""

from __future__ import annotations

import hashlib
import random


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
