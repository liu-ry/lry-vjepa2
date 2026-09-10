"""把 LeRobot 数据集转成训练器用的 JSONL manifest。

LeRobot 的相机/传感器 key 不固定，所以特征名必须显式指定。通过
``LeRobotDataset`` 逐帧解码，每个 episode 写成一个 ``.npy``。

Example::

    python -m app.vjepa_2_1.convert_lerobot \
      --repo-id user/my_dataset \
      --root /data/my_dataset \
      --vision-key observation.images.front \
      --tactile-key observation.images.gelsight \
      --state-key observation.state \
      --action-key action \
      --output /data/vjepa_manifest
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar(value: Any, default: int) -> int:
    array = _to_numpy(value) if value is not None else np.asarray(default)
    return int(array.reshape(-1)[0])


def _image_chw(value: Any) -> np.ndarray:
    """把单张图转成 CHW；向量特征原样返回。"""
    array = _to_numpy(value)
    if array.ndim != 3:
        return array
    if array.shape[0] in (1, 2, 3):
        return array
    if array.shape[-1] in (1, 2, 3):
        return np.transpose(array, (2, 0, 1))
    raise ValueError(f"cannot infer image channel dimension from shape {array.shape}")


def _stack(values: list[np.ndarray], key: str, episode: int) -> np.ndarray:
    if not values:
        raise ValueError(f"episode {episode} has no values for {key}")
    try:
        return np.stack(values, axis=0)
    except ValueError as exc:
        raise ValueError(f"episode {episode} has inconsistent shapes for {key}") from exc


def convert_dataset(
    repo_id: str,
    output: str | Path,
    vision_key: str,
    tactile_key: str,
    state_key: Optional[str] = None,
    action_key: Optional[str] = None,
    root: Optional[str | Path] = None,
    max_episodes: Optional[int] = None,
) -> Path:
    """解码 LeRobot 帧，返回生成的 JSONL manifest 路径。"""
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError:
        try:
            # LeRobot <=0.3 从这个模块导出该类。
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise ImportError(
                "LeRobot conversion requires the optional 'lerobot' package; "
                "install it in the environment used for data preparation"
            ) from exc

    kwargs = {"root": root} if root is not None else {}
    try:
        dataset = LeRobotDataset(repo_id, return_uint8=True, **kwargs)
    except TypeError:
        # 旧版 LeRobot 没有 return_uint8。
        dataset = LeRobotDataset(repo_id, **kwargs)

    records = defaultdict(lambda: {"frames": [], "vision": [], "tactile": [], "state": [], "action": []})
    for index in range(len(dataset)):
        row = dataset[index]
        episode = _scalar(row.get("episode_index"), 0)
        if max_episodes is not None and episode >= max_episodes:
            continue
        frame = _scalar(row.get("frame_index"), index)
        record = records[episode]
        record["frames"].append((frame, index))
        if vision_key not in row or tactile_key not in row:
            raise KeyError(f"row is missing required keys {vision_key!r} or {tactile_key!r}")
        record["vision"].append(_image_chw(row[vision_key]))
        record["tactile"].append(_image_chw(row[tactile_key]))
        if state_key is not None:
            if state_key not in row:
                raise KeyError(f"row is missing state key {state_key!r}")
            record["state"].append(_to_numpy(row[state_key]))
        if action_key is not None:
            if action_key not in row:
                raise KeyError(f"row is missing action key {action_key!r}")
            record["action"].append(_to_numpy(row[action_key]))

    output_path = Path(output)
    episode_path = output_path / "episodes"
    episode_path.mkdir(parents=True, exist_ok=True)
    manifest = output_path / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as stream:
        for episode in sorted(records):
            record = records[episode]
            order = np.argsort([frame for frame, _ in record["frames"]])
            vision = _stack([record["vision"][i] for i in order], vision_key, episode)
            tactile = _stack([record["tactile"][i] for i in order], tactile_key, episode)
            if len(vision) != len(tactile):
                raise ValueError(f"episode {episode} vision/tactile lengths differ")
            stem = f"episode_{episode:06d}"
            np.save(episode_path / f"{stem}_vision.npy", vision)
            np.save(episode_path / f"{stem}_tactile.npy", tactile)
            item = {
                "vision": f"episodes/{stem}_vision.npy",
                "tactile": f"episodes/{stem}_tactile.npy",
            }
            if state_key is not None:
                state = _stack([record["state"][i] for i in order], state_key, episode)
                if len(state) != len(vision):
                    raise ValueError(f"episode {episode} state length differs")
                np.save(episode_path / f"{stem}_state.npy", state)
                item["state"] = f"episodes/{stem}_state.npy"
            if action_key is not None:
                action = _stack([record["action"][i] for i in order], action_key, episode)
                if len(action) != len(vision):
                    raise ValueError(f"episode {episode} action length differs")
                np.save(episode_path / f"{stem}_action.npy", action)
                item["action"] = f"episodes/{stem}_action.npy"
            stream.write(json.dumps(item) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="LeRobot 仓库 id，例如 user/dataset")
    parser.add_argument("--root", default=None, help="本地 LeRobot 数据集根目录/缓存")
    parser.add_argument("--vision-key", required=True, help="相机特征 key")
    parser.add_argument("--tactile-key", required=True, help="触觉图像/向量特征 key")
    parser.add_argument("--state-key", default=None, help="可选本体状态特征 key")
    parser.add_argument("--action-key", default=None, help="可选动作特征 key")
    parser.add_argument("--output", required=True, help="npy 与 manifest.jsonl 的输出目录")
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()
    manifest = convert_dataset(**vars(args))
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
