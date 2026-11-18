import json

import numpy as np
import torch

from app.vjepa_2_1.data import RobotVisionTactileDataset
from app.vjepa_2_1.chain_data import ChainEpisodeDataset


def test_recorder_manifest_lazy_multipart_windows(tmp_path):
    frames = 12
    np.save(tmp_path / "vision.npy", np.zeros((frames, 3, 8, 8), dtype=np.uint8))
    np.save(tmp_path / "touch_left.npy", np.ones((frames, 3, 4, 4), dtype=np.uint8))
    np.save(tmp_path / "touch_right.npy", np.full((frames, 3, 4, 4), 2, dtype=np.uint8))
    np.save(tmp_path / "pose.npy", np.ones((frames, 7), dtype=np.float32))
    np.save(tmp_path / "gripper.npy", np.ones((frames, 1), dtype=np.float32))
    record = {
        "vision": "vision.npy",
        "tactile": ["touch_left.npy", "touch_right.npy"],
        "state": ["pose.npy", "gripper.npy"],
        "group": "task/episode_0",
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    dataset = RobotVisionTactileDataset(
        manifest, context=3, horizon=2, random_crop=False,
        require_state=True, vision_size=8, windows_per_episode=2,
    )
    assert len(dataset) == 2
    first, second = dataset[0], dataset[1]
    assert first["vision"].shape == (5, 3, 8, 8)
    assert first["tactile"].shape == (5, 3, 4, 8)
    assert first["state"].shape == (5, 8)
    assert torch.equal(first["tactile"][..., :4], second["tactile"][..., :4])

    context_only = RobotVisionTactileDataset(
        manifest, context=3, horizon=2, random_crop=False,
        require_state=True, vision_size=8, load_horizon=0,
    )
    sample = context_only[0]
    assert sample["vision"].shape == (3, 3, 8, 8)
    assert sample["tactile"].shape == (3, 3, 4, 8)


def test_contact_filter_requires_eligible_window(tmp_path):
    frames = 6
    np.save(tmp_path / "vision.npy", np.zeros((frames, 3, 4, 4), dtype=np.uint8))
    np.save(tmp_path / "touch.npy", np.zeros((frames, 3, 4, 4), dtype=np.uint8))
    np.save(
        tmp_path / "contact.npy",
        np.asarray([0, 0, 0, 1, 1, 0], dtype=np.uint8),
    )
    record = {
        "vision": "vision.npy",
        "tactile": "touch.npy",
        "contact": "contact.npy",
        "group": "task/episode_0",
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    dataset = RobotVisionTactileDataset(
        manifest,
        context=2,
        horizon=2,
        load_vision=False,
        load_state=False,
        require_contact=True,
        min_contact_frames=2,
        random_crop=True,
    )
    assert dataset.contact_window_starts(record).tolist() == [1, 2]
    sample = dataset[0]
    assert sample["tactile"].shape == (4, 3, 4, 4)
    assert sample["contact"].shape == (4,)
    assert sample["contact"].sum().item() == 2


def test_dense_windows_cover_episode_with_one_third_overlap(tmp_path):
    frames = 60
    np.save(tmp_path / "vision.npy", np.zeros((frames, 3, 4, 4), dtype=np.uint8))
    np.save(tmp_path / "touch.npy", np.zeros((frames, 3, 4, 4), dtype=np.uint8))
    np.save(tmp_path / "contact.npy", np.ones(frames, dtype=np.uint8))
    record = {
        "vision": "vision.npy",
        "tactile": "touch.npy",
        "contact": "contact.npy",
        "group": "task/episode_0",
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    dataset = RobotVisionTactileDataset(
        manifest,
        context=3,
        horizon=3,
        load_vision=False,
        load_state=False,
        require_contact=True,
        min_contact_frames=4,
        windows_per_episode=0,
        window_overlap=1.0 / 3.0,
    )
    starts = dataset._item_dense_starts(record)
    assert starts[0] == 0
    assert starts[-1] == frames - 6
    assert np.all(np.diff(starts[:-1]) == 4)
    assert abs((6 - 4) / 6 - 1.0 / 3.0) < 1e-9
    assert len(dataset) == len(starts)
    assert dataset[len(dataset) - 1]["tactile"].shape[0] == 6


def test_dense_cache_isolated_between_chains_in_same_group(tmp_path):
    frames = 40
    np.save(tmp_path / "vision.npy", np.zeros((frames, 3, 4, 4), dtype=np.uint8))
    for side in ("left", "right"):
        np.save(
            tmp_path / f"touch_{side}.npy",
            np.zeros((frames, 3, 4, 4), dtype=np.uint8),
        )
    np.save(
        tmp_path / "contact_left.npy",
        np.ones(frames, dtype=np.uint8),
    )
    np.save(
        tmp_path / "contact_right.npy",
        np.asarray([0] * 10 + [1] * 30, dtype=np.uint8),
    )
    records = [
        {
            "vision": "vision.npy",
            "tactile": f"touch_{side}.npy",
            "contact": f"contact_{side}.npy",
            "group": "task/episode_0",
        }
        for side in ("left", "right")
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    dataset = RobotVisionTactileDataset(
        manifest,
        context=2,
        horizon=4,
        load_vision=False,
        load_state=False,
        require_contact=True,
        min_contact_frames=3,
        windows_per_episode=0,
        window_overlap=1.0 / 3.0,
    )
    left_starts = dataset._item_dense_starts(records[0])
    right_starts = dataset._item_dense_starts(records[1])
    assert left_starts[0] == 0
    assert right_starts[0] == 7
    assert not np.array_equal(left_starts, right_starts)


def test_chain_dense_windows_use_overlapping_contact_starts(tmp_path):
    frames = 60
    np.save(tmp_path / "vision.npy", np.zeros((frames, 3, 4, 4), dtype=np.uint8))
    for side in ("left", "right"):
        np.save(
            tmp_path / f"tactile_{side}.npy",
            np.zeros((frames, 3, 4, 4), dtype=np.uint8),
        )
        np.save(
            tmp_path / f"contact_{side}.npy",
            np.ones(frames, dtype=np.uint8),
        )
        np.save(
            tmp_path / f"slip_{side}.npy",
            np.zeros(frames, dtype=np.uint8),
        )
    record = {
        "vision": "vision.npy",
        "vision_paths": ["vision.npy", "vision.npy"],
        "arm_valid": [True, True],
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
    dataset = ChainEpisodeDataset(
        manifest,
        context=3,
        horizon=3,
        load_vision=False,
        load_state=False,
        require_contact=True,
        min_contact_frames=4,
        min_chain_contact_frames=1,
        windows_per_episode=0,
        window_overlap=1.0 / 3.0,
    )
    starts = dataset._item_dense_starts(record)
    assert np.all(np.diff(starts[:-1]) == 4)
    assert len(dataset) == len(starts)


def test_stage_b_relaxed_windows_keep_no_contact_streams_valid(tmp_path):
    frames = 40
    context, horizon = 4, 6
    np.save(tmp_path / "vision.npy", np.zeros((frames, 3, 4, 4), dtype=np.uint8))
    contact = np.zeros(frames, dtype=np.uint8)
    contact[:20] = 1
    for side in ("left", "right"):
        np.save(
            tmp_path / f"tactile_{side}.npy",
            np.zeros((frames, 3, 4, 4), dtype=np.uint8),
        )
        np.save(
            tmp_path / f"contact_{side}.npy",
            contact,
        )
        np.save(
            tmp_path / f"slip_{side}.npy",
            np.zeros(frames, dtype=np.uint8),
        )
    record = {
        "vision": "vision.npy",
        "vision_paths": ["vision.npy", "vision.npy"],
        "arm_valid": [True, True],
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


def test_stage_b_mixed_and_stratified_window_sampling(tmp_path):
    frames = 40
    context, horizon = 4, 6
    np.save(tmp_path / "vision.npy", np.zeros((frames, 3, 4, 4), dtype=np.uint8))
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


def test_single_hand_record_is_padded_and_masked(tmp_path):
    frames = 6
    np.save(tmp_path / "vision.npy", np.zeros((frames, 3, 4, 4), dtype=np.uint8))
    np.save(
        tmp_path / "tactile_left.npy",
        np.ones((frames, 3, 4, 4), dtype=np.uint8),
    )
    np.save(
        tmp_path / "contact_left.npy",
        np.asarray([1, 1, 1, 1, 1, 1], dtype=np.uint8),
    )
    record = {
        "vision": "vision.npy",
        "tactile": "tactile_left.npy",
        "contact": "contact_left.npy",
        "group": "task/episode_single",
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    dataset = RobotVisionTactileDataset(
        manifest,
        context=2,
        horizon=2,
        load_vision=False,
        load_state=False,
        random_crop=False,
        require_contact=True,
        min_contact_frames=4,
        min_hand_contact_frames=1,
        num_hands=2,
    )
    sample = dataset[0]
    assert sample["tactile_left"].shape == (4, 3, 4, 4)
    assert torch.count_nonzero(sample["tactile_right"]) == 0
    assert sample["hand_valid"].tolist() == [True, False]
    assert sample["contact"].shape == (4, 2)
    assert sample["contact"][:, 1].sum().item() == 0


def test_dual_record_with_inactive_hand_masks_that_hand(tmp_path):
    frames = 6
    np.save(tmp_path / "vision.npy", np.zeros((frames, 3, 4, 4), dtype=np.uint8))
    np.save(
        tmp_path / "tactile_left.npy",
        np.ones((frames, 3, 4, 4), dtype=np.uint8),
    )
    np.save(
        tmp_path / "tactile_right.npy",
        np.zeros((frames, 3, 4, 4), dtype=np.uint8),
    )
    np.save(
        tmp_path / "contact_left.npy",
        np.ones(frames, dtype=np.uint8),
    )
    np.save(
        tmp_path / "contact_right.npy",
        np.zeros(frames, dtype=np.uint8),
    )
    record = {
        "vision": "vision.npy",
        "tactile": ["tactile_left.npy", "tactile_right.npy"],
        "contact": ["contact_left.npy", "contact_right.npy"],
        "group": "task/episode_dual",
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    dataset = RobotVisionTactileDataset(
        manifest,
        context=2,
        horizon=2,
        load_vision=False,
        load_state=False,
        random_crop=False,
        require_contact=True,
        min_contact_frames=4,
        min_hand_contact_frames=1,
        num_hands=2,
    )
    assert dataset.contact_window_starts(record).tolist() == [0, 1, 2]
    sample = dataset[0]
    assert sample["hand_valid"].tolist() == [True, False]
