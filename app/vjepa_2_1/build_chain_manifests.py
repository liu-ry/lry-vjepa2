"""Expand arm-level recorder manifests into four independent tactile chains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CHAIN_ORDER = (
    ("left", "left"),
    ("left", "right"),
    ("right", "left"),
    ("right", "right"),
)


def _arm_from_stream(stream: str) -> str | None:
    name = Path(stream).name.lower()
    if "right" in name:
        return "right"
    if "left" in name:
        return "left"
    return None


def _sensor_from_path(path: str) -> str:
    name = Path(path).name.lower()
    return "right" if "right" in name else "left"


def _absolute(source_root: Path, value):
    if value is None:
        return None
    if isinstance(value, list):
        return [_absolute(source_root, item) for item in value]
    path = Path(value)
    return str(path if path.is_absolute() else (source_root / path).resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.manifest).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [
        json.loads(line)
        for line in source.open(encoding="utf-8")
        if line.strip()
    ]

    chain_rows = []
    groups: dict[str, dict[tuple[str, str], dict]] = {}
    for record in rows:
        tactile = record.get("tactile")
        paths = [tactile] if isinstance(tactile, str) else list(tactile or [])
        if not paths:
            continue
        arm = _arm_from_stream(record.get("stream", ""))
        if arm is None:
            continue
        sensors = record.get("tactile_sides")
        if sensors is None:
            sensors = [_sensor_from_path(path) for path in paths]
        contact = record.get("contact")
        contact_paths = (
            [contact] if isinstance(contact, str) else list(contact or [])
        )
        slip = record.get("slip")
        slip_paths = [slip] if isinstance(slip, str) else list(slip or [])
        if len(contact_paths) != len(paths) or len(slip_paths) != len(paths):
            raise ValueError(
                f"chain labels do not align for {record.get('stream')!r}"
            )
        group = str(record.get("group", record.get("stream", "")))
        groups.setdefault(group, {})
        for path, sensor, contact_path, slip_path in zip(
            paths, sensors, contact_paths, slip_paths
        ):
            chain_id = f"{arm}_hand_{sensor}_sensor"
            chain = {
                "vision": _absolute(source.parent, record.get("vision")),
                "tactile": _absolute(source.parent, path),
                "contact": _absolute(source.parent, contact_path),
                "slip": _absolute(source.parent, slip_path),
                "group": group,
                "task": record.get("task", "?"),
                "stream": record.get("stream"),
                "arm_id": arm,
                "sensor_id": sensor,
                "chain_id": chain_id,
            }
            if "state" in record:
                chain["state"] = _absolute(source.parent, record["state"])
            chain_rows.append(chain)
            groups[group][(arm, sensor)] = chain

    chain_manifest = output_root / "chains.jsonl"
    with chain_manifest.open("w", encoding="utf-8") as stream:
        for row in chain_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    episode_rows = []
    for group, chains in sorted(groups.items()):
        available = [chains.get(key) for key in CHAIN_ORDER]
        if not any(available):
            continue
        template = next(item for item in available if item is not None)
        arm_records = {}
        for item in available:
            if item is not None:
                arm_records.setdefault(item["arm_id"], item)
        episode = {
            "vision": template.get("vision"),
            "vision_paths": [
                arm_records.get("left", {}).get("vision"),
                arm_records.get("right", {}).get("vision"),
            ],
            "arm_valid": ["left" in arm_records, "right" in arm_records],
            "group": group,
            "task": template.get("task", "?"),
            "chain_ids": [f"{arm}_hand_{sensor}_sensor" for arm, sensor in CHAIN_ORDER],
            "chain_valid": [item is not None for item in available],
            "tactile": [item["tactile"] if item is not None else None for item in available],
            "contact": [item["contact"] if item is not None else None for item in available],
            "slip": [item["slip"] if item is not None else None for item in available],
            "state_paths": [
                arm_records.get("left", {}).get("state"),
                arm_records.get("right", {}).get("state"),
            ],
        }
        if "state" in template:
            episode["state"] = template["state"]
        episode_rows.append(episode)

    episode_manifest = output_root / "episodes_4chain.jsonl"
    with episode_manifest.open("w", encoding="utf-8") as stream:
        for row in episode_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "source_records": len(rows),
        "chain_records": len(chain_rows),
        "episode_records": len(episode_rows),
        "one_chain": sum(sum(row["chain_valid"]) == 1 for row in episode_rows),
        "two_chains": sum(sum(row["chain_valid"]) == 2 for row in episode_rows),
        "three_chains": sum(sum(row["chain_valid"]) == 3 for row in episode_rows),
        "four_chains": sum(sum(row["chain_valid"]) == 4 for row in episode_rows),
        "chain_manifest": str(chain_manifest),
        "episode_manifest": str(episode_manifest),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
