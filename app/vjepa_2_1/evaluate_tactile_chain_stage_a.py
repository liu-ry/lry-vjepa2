"""Evaluate Stage-A per-horizon latent forecast error."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.vjepa_2_1.train_tactile_chain_stage_a import (
    ChainStageAModule,
    build_loaders,
)


def evaluate(module, loader, split_index: int, max_batches: int):
    records = []
    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            module(batch)
            diagnostics = module._last_diagnostics
            prediction = diagnostics["predicted_future"]
            target = diagnostics["target_future"]
            base = diagnostics["base"][:, None]
            scale = 0.01
            error = F.smooth_l1_loss(
                (prediction - base) / scale,
                (target - base) / scale,
                reduction="none",
                beta=0.2,
            ).mean(dim=(-1, -2))
            copy = F.smooth_l1_loss(
                torch.zeros_like(target),
                (target - base) / scale,
                reduction="none",
                beta=0.2,
            ).mean(dim=(-1, -2))
            records.append(
                {
                    "error": error.sum(dim=0),
                    "copy": copy.sum(dim=0),
                    "normalized_mse": (
                        F.normalize(prediction, dim=-1)
                        - F.normalize(target, dim=-1)
                    ).square().mean(dim=(-1, -2)).sum(dim=0),
                    "cosine": F.cosine_similarity(
                        prediction, target, dim=-1
                    ).mean(dim=-1).sum(dim=0),
                    "count": prediction.shape[0],
                }
            )
            if max_batches and index >= max_batches:
                break
    error = torch.stack([record["error"] for record in records]).sum(0)
    copy = torch.stack([record["copy"] for record in records]).sum(0)
    mse = torch.stack(
        [record["normalized_mse"] for record in records]
    ).sum(0)
    cosine = torch.stack([record["cosine"] for record in records]).sum(0)
    count = sum(record["count"] for record in records)
    relative = (error + 1e-8) / (copy + 1e-8)
    return {
        "split_index": split_index,
        "samples": int(count),
        "horizon_offsets": list(range(1, len(relative) + 1)),
        "relative": [float(value) for value in relative],
        "normalized_mse": [float(value / count) for value in mse],
        "cosine": [float(value / count) for value in cosine],
        "aggregate_relative": float(error.sum() / copy.sum().clamp_min(1e-8)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--attention", required=True)
    parser.add_argument("--latent-grid", required=True)
    parser.add_argument(
        "--objective",
        choices=("future_forecast", "clip_masked"),
        default="future_forecast",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-batches", type=int, default=0)
    parser.add_argument("--test-batches", type=int, default=0)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg["encoder_training"]["temporal_attention"] = args.attention
    cfg["encoder_training"]["stage_a_objective"] = args.objective
    height, width = args.latent_grid.lower().split("x", 1)
    cfg["encoder_training"]["latent_spatial_grid"] = [
        int(height),
        int(width),
    ]
    module = ChainStageAModule(cfg, "cuda:0")
    state = torch.load(
        args.checkpoint, map_location="cuda:0", weights_only=False
    )
    module.encoder.load_state_dict(state["encoder"])
    module.encoder_target.load_state_dict(
        state.get("encoder_target", state["encoder"])
    )
    module.predictor.load_state_dict(state["predictor"])
    module.contact_probe.load_state_dict(state["contact_probe"])
    module.slip_probe.load_state_dict(state["slip_probe"])
    module.eval()
    _, val_loader, test_loader, split = build_loaders(
        cfg, batch_size=1, rank=0, world_size=1
    )
    report = {
        "validation": evaluate(
            module, val_loader, 1, args.val_batches
        ),
        "test": evaluate(
            module, test_loader, 2, args.test_batches
        ),
        "split": split,
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
