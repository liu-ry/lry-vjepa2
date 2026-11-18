import json

import numpy as np

from tac_jepa.data.chain import ChainEpisodeDataset


def _write_episode(tmp_path):
    frames = 40
    context, horizon = 4, 6
    np.save(
        tmp_path / "vision.npy",
        np.zeros((frames, 3, 4, 4), dtype=np.uint8),
    )
    contact = np.zeros(frames, dtype=np.uint8)
    contact[:20] = 1
    for side in ("left", "right"):
        np.save(
            tmp_path / f"tactile_{side}.npy",
            np.zeros((frames, 3, 4, 4), dtype=np.uint8),
        )
        np.save(tmp_path / f"contact_{side}.npy", contact)
        np.save(
            tmp_path / f"slip_{side}.npy",
            np.zeros(frames, dtype=np.uint8),
        )
    record = {
        "vision": "vision.npy",
        "vision_paths": ["vision.npy", "vision.npy"],
        "arm_valid": [True, True],
        "chain_valid": [True, True, True, True],
        "tactile": [
            "tactile_left.npy",
            "tactile_right.npy",
            "tactile_left.npy",
            "tactile_right.npy",
        ],
        "contact": [
            "contact_left.npy",
            "contact_right.npy",
            "contact_left.npy",
            "contact_right.npy",
        ],
        "slip": [
            "slip_left.npy",
            "slip_right.npy",
            "slip_left.npy",
            "slip_right.npy",
        ],
        "group": "task/episode_0",
    }
    manifest = tmp_path / "episodes.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return record, manifest, context, horizon


def test_natural_windows_keep_no_contact_streams_valid(tmp_path):
    record, manifest, context, horizon = _write_episode(tmp_path)
    dataset = ChainEpisodeDataset(
        manifest,
        context=context,
        horizon=horizon,
        load_vision=False,
        load_state=False,
        require_contact=False,
        window_sampling="all",
        windows_per_episode=0,
        window_overlap=1.0 / 3.0,
    )
    starts = dataset._item_dense_starts(record)
    assert starts[-1] > 20
    sample = dataset[len(dataset) - 1]
    assert sample["contact"].sum().item() == 0
    assert sample["chain_valid"].all()

    categories = dataset._categorized_window_starts(record)
    assert len(categories["ongoing"]) > 0
    assert len(categories["end"]) > 0
    assert len(categories["no_contact"]) > 0


def test_mixed_and_stratified_windows(tmp_path):
    record, manifest, context, horizon = _write_episode(tmp_path)
    mixed = ChainEpisodeDataset(
        manifest,
        context=context,
        horizon=horizon,
        load_vision=False,
        load_state=False,
        require_contact=False,
        window_sampling="mixed",
        no_contact_fraction=0.4,
        windows_per_episode=0,
        window_overlap=1.0 / 3.0,
    )
    starts = mixed._item_dense_starts(record)
    categories = mixed._categorized_window_starts(record)
    start_count = set(starts.tolist())
    no_contact_count = len(
        start_count & set(categories["no_contact"].tolist())
    )
    assert no_contact_count > 0
    assert no_contact_count / len(starts) <= 0.4 + 1e-9

    stratified = ChainEpisodeDataset(
        manifest,
        context=context,
        horizon=horizon,
        load_vision=False,
        load_state=False,
        require_contact=False,
        window_sampling="stratified",
        windows_per_episode=2,
        random_crop=False,
    )
    sums = [
        int(stratified[index]["contact"].sum().item())
        for index in range(len(stratified))
    ]
    assert max(sums) > 0
    assert min(sums) == 0

