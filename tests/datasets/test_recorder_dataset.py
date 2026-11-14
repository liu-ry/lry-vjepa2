import json

import numpy as np
import torch

from app.vjepa_2_1.data import RobotVisionTactileDataset


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
