"""Run Stage A and Stage B sequentially with automatic checkpoint handoff."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    index = 1
    while path.with_name(f"{path.name}({index})").exists():
        index += 1
    return path.with_name(f"{path.name}({index})")


def _run(command: list[str], *, dry_run: bool) -> None:
    print("+ " + shlex.join(command), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not existing
        else str(PROJECT_ROOT) + os.pathsep + existing
    )
    subprocess.run(command, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-config", required=True)
    parser.add_argument("--stage-b-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention", default="per_frame_causal")
    parser.add_argument("--latent-grid", default="7x7")
    parser.add_argument("--stage-a-max-steps", type=int, default=None)
    parser.add_argument("--stage-b-max-steps", type=int, default=None)
    parser.add_argument("--stage-a-batch-size", type=int, default=None)
    parser.add_argument("--stage-b-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    stage_a_output = _unique_dir(output_root / "stage_a")
    stage_b_output = _unique_dir(output_root / "stage_b")
    python = sys.executable

    stage_a = [
        python,
        "-m",
        "tac_jepa.training.stage_a",
        "--config",
        str(Path(args.stage_a_config).resolve()),
        "--output",
        str(stage_a_output),
        "--device",
        args.device,
    ]
    if args.stage_a_max_steps is not None:
        stage_a.extend(("--max-steps", str(args.stage_a_max_steps)))
    if args.stage_a_batch_size is not None:
        stage_a.extend(("--batch-size", str(args.stage_a_batch_size)))
    if args.num_workers is not None:
        stage_a.extend(("--num-workers", str(args.num_workers)))
    _run(stage_a, dry_run=args.dry_run)

    encoder_checkpoint = stage_a_output / "best.pt"
    stage_b = [
        python,
        "-m",
        "tac_jepa.training.stage_b",
        "--config",
        str(Path(args.stage_b_config).resolve()),
        "--encoder-checkpoint",
        str(encoder_checkpoint),
        "--output",
        str(stage_b_output),
        "--device",
        args.device,
        "--attention",
        args.attention,
        "--latent-grid",
        args.latent_grid,
    ]
    if args.stage_b_max_steps is not None:
        stage_b.extend(("--max-steps", str(args.stage_b_max_steps)))
    if args.stage_b_batch_size is not None:
        stage_b.extend(("--batch-size", str(args.stage_b_batch_size)))
    if args.num_workers is not None:
        stage_b.extend(("--num-workers", str(args.num_workers)))
    _run(stage_b, dry_run=args.dry_run)

    print(f"Stage A output: {stage_a_output}")
    print(f"Stage B output: {stage_b_output}")


if __name__ == "__main__":
    main()
