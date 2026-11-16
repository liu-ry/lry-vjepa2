"""One-command dataset conversion and Tac-JEPA manifest preparation."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parent


def _run(command: list[str]) -> None:
    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        required=True,
        choices=("recorder", "manifest"),
        help=(
            "Input format for the full pipeline. Use 'manifest' for an "
            "arm-level manifest outside recorder layout."
        ),
    )
    parser.add_argument("--output", required=True, help="Prepared dataset root")
    parser.add_argument("--root", help="Raw recorder dataset root")
    parser.add_argument(
        "--manifest",
        help="Existing arm-level JSONL manifest for --format manifest",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--allow-missing-state", action="store_true")
    parser.add_argument("--include-invalid", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--contact-threshold", type=float, default=0.025)
    parser.add_argument("--slip-threshold", type=float, default=0.012)
    parser.add_argument(
        "--skip-contacts",
        action="store_true",
        help="Skip contact/slip generation when the input manifest already has labels.",
    )
    parser.add_argument(
        "--skip-chains",
        action="store_true",
        help="Stop after contact manifest generation.",
    )
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    if args.format == "manifest":
        if not args.manifest:
            parser.error("--format manifest requires --manifest")
        raw_manifest = Path(args.manifest).resolve()
    else:
        if not args.root:
            parser.error("--format recorder requires --root")
        converted = output / "converted"
        command = [
            python,
            str(TOOLS / "convert_data.py"),
            "--format",
            args.format,
            "--output",
            str(converted),
        ]
        optional = {
            "--root": args.root,
            "--max-episodes": args.max_episodes,
        }
        for flag, value in optional.items():
            if value is not None:
                command.extend((flag, str(value)))
        for flag, enabled in (
            ("--allow-missing-state", args.allow_missing_state),
            ("--include-invalid", args.include_invalid),
        ):
            if enabled:
                command.append(flag)
        _run(command)
        raw_manifest = converted / "manifest.jsonl"

    if args.skip_contacts:
        contact_manifest = raw_manifest
    else:
        contact_root = output / "contact_manifest"
        _run(
            [
                python,
                str(TOOLS / "build_contacts.py"),
                "--manifest",
                str(raw_manifest),
                "--output",
                str(contact_root),
                "--workers",
                str(args.workers),
                "--contact-threshold",
                str(args.contact_threshold),
                "--slip-threshold",
                str(args.slip_threshold),
            ]
        )
        contact_manifest = contact_root / "manifest.jsonl"

    if not args.skip_chains:
        chain_root = output / "chains"
        _run(
            [
                python,
                str(TOOLS / "build_chains.py"),
                "--manifest",
                str(contact_manifest),
                "--output",
                str(chain_root),
            ]
        )
        print(f"Stage A manifest: {chain_root / 'chains.jsonl'}")
        print(
            "Stage B manifest: "
            f"{chain_root / 'episodes_4chain.jsonl'}"
        )
    else:
        print(f"Contact manifest: {contact_manifest}")


if __name__ == "__main__":
    main()
