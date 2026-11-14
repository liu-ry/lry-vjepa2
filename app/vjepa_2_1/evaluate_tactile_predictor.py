"""Evaluate a trained causal tactile predictor on held-out episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.vjepa_2_1.train_tactile_two_stage import (
    PredictorStage,
    _apply_overrides,
    _build_loaders,
    _evaluate_predictor,
    _load_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    cfg = _apply_overrides(_load_config(args.config), args.overrides)
    section = cfg["predictor_training"]
    trainer = PredictorStage(cfg, section, args.device, args.encoder_checkpoint)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    trainer.predictor.load_state_dict(state["predictor"])
    trainer.contact_probe.load_state_dict(state["contact_probe"])
    trainer.slip_probe.load_state_dict(state["slip_probe"])
    if state.get("encoder") is not None:
        trainer.encoder.load_state_dict(state["encoder"])
    if trainer.fine_tune_encoder and state.get("encoder_target") is not None:
        trainer.encoder_target.load_state_dict(state["encoder_target"])

    batch_size = int(section.get("batch_size", cfg.get("batch_size", 4)))
    _, val_loader, test_loader, split = _build_loaders(
        cfg,
        stage="predictor",
        batch_size=batch_size,
        distributed=False,
        rank=0,
        world_size=1,
        max_records=args.max_records,
    )
    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "step": int(state.get("step", -1)),
        "validation": _evaluate_predictor(
            trainer, val_loader, args.max_val_batches
        ),
        "test": _evaluate_predictor(
            trainer, test_loader, args.max_val_batches
        ),
        "split": {
            key: split[key]
            for key in (
                "n_train_records",
                "n_val_records",
                "n_test_records",
                "n_train_windows",
                "n_val_windows",
                "n_test_windows",
            )
        },
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
