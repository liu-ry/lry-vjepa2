"""Build contact masks by comparing tactile frames with each stream's first frame."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def _fill_short_false_runs(mask: np.ndarray, max_gap: int) -> np.ndarray:
    result = mask.astype(bool).copy()
    true_idx = np.flatnonzero(result)
    if len(true_idx) < 2:
        return result.astype(np.uint8)
    for left, right in zip(true_idx[:-1], true_idx[1:]):
        if 1 < right - left <= max_gap + 1:
            result[left + 1:right] = True
    return result.astype(np.uint8)


def _analyse_video(path: str, contact_threshold: float, slip_threshold: float):
    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise OSError(f"cannot open {path}")
    ok, first = capture.read()
    if not ok:
        capture.release()
        raise ValueError(f"cannot decode first frame from {path}")
    first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    # OpenCV uses (width, height). The source tactile videos are 300x200
    # (height x width), so 96x64 preserves their 3:2 geometry.
    first = cv2.resize(first, (64, 96), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    previous = first
    contact, slip = [0], [0]
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 96), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        contact_score = float(np.percentile(np.abs(gray - first), 95))
        slip_score = float(np.percentile(np.abs(gray - previous), 95))
        contact.append(int(contact_score >= contact_threshold))
        slip.append(int(slip_score >= slip_threshold))
        previous = gray
    capture.release()
    contact = _fill_short_false_runs(np.asarray(contact), max_gap=3)
    slip = _fill_short_false_runs(np.asarray(slip), max_gap=1)
    return contact, slip


def _process_record(job):
    index, record, output_dir, contact_threshold, slip_threshold = job
    tactile = record.get("tactile")
    paths = (
        [tactile]
        if isinstance(tactile, str)
        else list(tactile) if isinstance(tactile, list) else []
    )
    if not 1 <= len(paths) <= 2:
        return None
    configured_sides = record.get("tactile_sides")
    if configured_sides is not None:
        sides = [str(side).lower() for side in configured_sides]
        if len(sides) != len(paths) or any(side not in ("left", "right") for side in sides):
            return None
    else:
        sides = []
        for path in paths:
            name = Path(path).name.lower()
            sides.append("right" if "right" in name else "left")
    digest = hashlib.sha1(
        f"{index}:{record.get('group')}:{record.get('stream')}".encode("utf-8")
    ).hexdigest()[:20]
    contact_paths, slip_paths = [], []
    fractions = {"left": 0.0, "right": 0.0}
    for side, path in zip(sides, paths):
        contact, slip = _analyse_video(
            path, contact_threshold=contact_threshold, slip_threshold=slip_threshold
        )
        fractions[side] = float(contact.mean())
        contact_path = output_dir / "contact" / f"{digest}_{side}.npy"
        slip_path = output_dir / "slip" / f"{digest}_{side}.npy"
        contact_path.parent.mkdir(parents=True, exist_ok=True)
        slip_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(contact_path, contact)
        np.save(slip_path, slip)
        contact_paths.append(str(contact_path.relative_to(output_dir)))
        slip_paths.append(str(slip_path.relative_to(output_dir)))
    if max(fractions.values()) <= 0:
        return None
    result = dict(record)
    result["contact"] = contact_paths
    result["slip"] = slip_paths
    result["tactile_sides"] = sides
    result["contact_fraction"] = max(fractions.values())
    result["contact_fraction_left"] = fractions["left"]
    result["contact_fraction_right"] = fractions["right"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--contact-threshold", type=float, default=0.025)
    parser.add_argument("--slip-threshold", type=float, default=0.012)
    args = parser.parse_args()
    source = Path(args.manifest).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in source.open() if line.strip()]
    jobs = [
        (
            index,
            record,
            output,
            float(args.contact_threshold),
            float(args.slip_threshold),
        )
        for index, record in enumerate(rows)
    ]
    contact_rows = []
    fractions = []
    tasks = Counter()
    with mp.Pool(max(1, int(args.workers))) as pool:
        for processed, result in enumerate(
            pool.imap_unordered(_process_record, jobs, chunksize=1), start=1
        ):
            if result is not None:
                contact_rows.append(result)
                fractions.append(float(result["contact_fraction"]))
                tasks[result.get("task", "?")] += 1
            if processed % 250 == 0:
                print(
                    f"processed {processed}/{len(jobs)} contact_records="
                    f"{len(contact_rows)}",
                    flush=True,
                )
    contact_rows.sort(key=lambda row: (row.get("group", ""), row.get("stream", "")))
    manifest = output / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as stream:
        for row in contact_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = {
        "source_manifest": str(source),
        "contact_threshold": float(args.contact_threshold),
        "slip_threshold": float(args.slip_threshold),
        "source_records": len(rows),
        "contact_records": len(contact_rows),
        "contact_groups": len({row["group"] for row in contact_rows}),
        "contact_fraction_mean": float(np.mean(fractions)) if fractions else 0.0,
        "contact_fraction_median": float(np.median(fractions)) if fractions else 0.0,
        "fraction_ge_0_1": int(sum(value >= 0.1 for value in fractions)),
        "fraction_ge_0_25": int(sum(value >= 0.25 for value in fractions)),
        "tasks": dict(sorted(tasks.items())),
    }
    (output / "stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(stats, indent=2, sort_keys=True), flush=True)
    print(f"wrote {manifest}", flush=True)


if __name__ == "__main__":
    main()
