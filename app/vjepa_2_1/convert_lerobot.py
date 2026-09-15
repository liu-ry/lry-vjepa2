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


def _decode_mp4(path: Path) -> np.ndarray:
    """Decode an mp4 into uint8 TCHW (BGR is converted to RGB)."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("UMI conversion requires opencv-python") from exc
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
    capture.release()
    if not frames:
        raise ValueError(f"could not decode any frames from {path}")
    return np.stack(frames, axis=0)


def convert_umi_dataset(root: str | Path, output: str | Path,
                        max_episodes: Optional[int] = None,
                        include_state: bool = False) -> Path:
    """Convert the VT-UMI directory layout into the trainer JSONL manifest.

    Each ``episode_*/left_hand`` directory must contain ``rgb.mp4``,
    ``tactile_left.mp4`` and ``tactile_right.mp4``.  The two finger videos are
    concatenated horizontally (preserving both sensors) into one 3-channel
    tactile frame.  When requested, ``vio_pose.npy`` and ``gripper.npy`` are
    concatenated along the feature dimension and exported as state.
    """
    root, output_path = Path(root), Path(output)
    episode_path = output_path / "episodes"
    episode_path.mkdir(parents=True, exist_ok=True)
    episodes = sorted(root.glob("episode_*/left_hand"),
                      key=lambda p: int(p.parent.name.split("_")[-1]))
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    if not episodes:
        raise ValueError(f"no episode_*/left_hand directories found under {root}")
    total_episodes = len(episodes)
    print(f"[convert] found {total_episodes} episode(s)", flush=True)
    manifest = output_path / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as stream:
        for number, hand in enumerate(episodes, start=1):
            print(
                f"[convert] processing episode {number}/{total_episodes}: "
                f"{hand.parent.name}",
                flush=True,
            )
            vision = _decode_mp4(hand / "rgb.mp4")
            tactile_left = _decode_mp4(hand / "tactile_left.mp4")
            tactile_right = _decode_mp4(hand / "tactile_right.mp4")
            count = min(len(vision), len(tactile_left), len(tactile_right))
            if count == 0:
                print(f"[convert] skipped empty episode: {hand.parent.name}", flush=True)
                continue
            # Keep both gripper fingers: [T,C,H,W] -> [T,C,H,2W].
            if tactile_left.shape[1:] != tactile_right.shape[1:]:
                raise ValueError(f"episode {hand.parent.name} tactile frame shapes differ")
            tactile = np.concatenate((tactile_left[:count], tactile_right[:count]), axis=-1)
            vision = vision[:count]
            # Keep the generated UMI episode ids zero-based for compatibility
            # with existing manifests; ``number`` is one-based for progress.
            stem = f"episode_{number - 1:06d}"
            np.save(episode_path / f"{stem}_vision.npy", vision)
            np.save(episode_path / f"{stem}_tactile.npy", tactile)
            item = {"vision": f"episodes/{stem}_vision.npy",
                    "tactile": f"episodes/{stem}_tactile.npy"}
            if include_state:
                pose_path, gripper_path = hand / "vio_pose.npy", hand / "gripper.npy"
                for state_path in (pose_path, gripper_path):
                    if not state_path.exists():
                        raise FileNotFoundError(
                            f"--include-state requested but missing {state_path}"
                        )
                pose = np.asarray(np.load(pose_path))[:count]
                gripper = np.asarray(np.load(gripper_path))[:count]
                if pose.ndim == 1:
                    pose = pose[:, None]
                if gripper.ndim == 1:
                    gripper = gripper[:, None]
                if len(pose) != count or len(gripper) != count:
                    raise ValueError(f"episode {hand.parent.name} state length differs")
                if pose.ndim != 2 or gripper.ndim != 2:
                    raise ValueError(f"episode {hand.parent.name} state arrays must be 1D/2D")
                state = np.concatenate((pose, gripper), axis=-1)
                np.save(episode_path / f"{stem}_state.npy", state)
                item["state"] = f"episodes/{stem}_state.npy"
            stream.write(json.dumps(item) + "\n")
    return manifest


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
    total_episodes = len(records)
    print(f"[convert] found {total_episodes} episode(s)", flush=True)
    with manifest.open("w", encoding="utf-8") as stream:
        for number, episode in enumerate(sorted(records), start=1):
            print(f"[convert] processing episode {number}/{total_episodes}: {episode}", flush=True)
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
    parser.add_argument("--format", choices=("lerobot", "umi"), default="lerobot",
                        help="输入格式；VT-UMI 数据使用 umi")
    parser.add_argument("--repo-id", help="LeRobot 仓库 id，例如 user/dataset")
    parser.add_argument("--root", default=None, help="本地 LeRobot 数据集根目录/缓存")
    parser.add_argument("--vision-key", default=None, help="相机特征 key (LeRobot)")
    parser.add_argument("--tactile-key", default=None, help="触觉图像/向量特征 key (LeRobot)")
    parser.add_argument("--state-key", default=None, help="可选本体状态特征 key")
    parser.add_argument("--action-key", default=None, help="可选动作特征 key")
    parser.add_argument("--output", required=True, help="npy 与 manifest.jsonl 的输出目录")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--include-state", action="store_true",
                        help="UMI: 拼接 left_hand/vio_pose.npy + gripper.npy 作为 state")
    args = parser.parse_args()
    if args.format == "umi":
        if not args.root:
            parser.error("--format umi requires --root DATASET_ROOT")
        manifest = convert_umi_dataset(args.root, args.output, args.max_episodes,
                                       args.include_state)
    else:
        if not args.repo_id:
            parser.error("--format lerobot requires --repo-id")
        if not args.vision_key or not args.tactile_key:
            parser.error("LeRobot conversion requires --vision-key and --tactile-key")
        manifest = convert_dataset(
            repo_id=args.repo_id, root=args.root, vision_key=args.vision_key,
            tactile_key=args.tactile_key, state_key=args.state_key,
            action_key=args.action_key, output=args.output,
            max_episodes=args.max_episodes)
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
