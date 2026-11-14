"""Evaluate a two-stage tactile encoder checkpoint on held-out episodes."""

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
    EncoderStage,
    _apply_overrides,
    _build_loaders,
    _encode_indices,
    _evaluate_encoder,
    _image_tensor_to_float,
    _load_config,
)


@torch.no_grad()
def latent_dynamics(
    cfg: dict,
    trainer: EncoderStage,
    loader,
    max_batches: int,
) -> dict[str, float]:
    trainer.encoder.eval()
    context = int(cfg["context"])
    event_offsets = [int(x) for x in cfg["event_offsets"]]
    planning_offsets = [int(x) for x in cfg["planning_offsets"]]
    event_indices = [context + x - 1 for x in event_offsets]
    planning_indices = [context + x - 1 for x in planning_offsets]
    sums, count = {}, 0
    for batch in loader:
        left = _image_tensor_to_float(batch["tactile_left"].to(trainer.device))
        right = _image_tensor_to_float(batch["tactile_right"].to(trainer.device))
        ctx = _encode_indices(
            trainer.encoder, left, right, list(range(context)), True
        )
        event = _encode_indices(
            trainer.encoder, left, right, event_indices, True
        )
        planning = _encode_indices(
            trainer.encoder, left, right, planning_indices, True
        )
        current = ctx[:, -1].unsqueeze(1)
        for name, target in (("event", event), ("planning", planning)):
            reference = current.expand_as(target)
            l2 = (target - reference).square().mean()
            cosine = (
                1.0
                - (
                    torch.nn.functional.normalize(target, dim=-1)
                    * torch.nn.functional.normalize(reference, dim=-1)
                ).sum(dim=-1)
            ).mean()
            delta = target - reference
            delta_rms = delta.square().mean().sqrt()
            target_rms = target.square().mean().sqrt()
            sums[f"{name}_l2_copy"] = sums.get(f"{name}_l2_copy", 0.0) + float(l2)
            sums[f"{name}_cosine_change"] = (
                sums.get(f"{name}_cosine_change", 0.0) + float(cosine)
            )
            sums[f"{name}_delta_rms"] = (
                sums.get(f"{name}_delta_rms", 0.0) + float(delta_rms)
            )
            sums[f"{name}_target_rms"] = (
                sums.get(f"{name}_target_rms", 0.0) + float(target_rms)
            )
        count += 1
        if max_batches and count >= max_batches:
            break
    return {key: value / max(1, count) for key, value in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=24)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    cfg = _apply_overrides(_load_config(args.config), args.overrides)
    trainer = EncoderStage(cfg, args.device)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    trainer.encoder.load_state_dict(state["encoder"])
    if (
        trainer.encoder_target is not None
        and state.get("encoder_target") is not None
    ):
        trainer.encoder_target.load_state_dict(state["encoder_target"])
    if (
        trainer.latent_predictor is not None
        and state.get("latent_predictor") is not None
    ):
        trainer.latent_predictor.load_state_dict(state["latent_predictor"])
    trainer.decoder.load_state_dict(state["decoder"])
    if state.get("contact_probe") is not None:
        trainer.contact_probe.load_state_dict(state["contact_probe"])
        trainer.slip_probe.load_state_dict(state["slip_probe"])
    _, val_loader, test_loader, _ = _build_loaders(
        cfg,
        stage="encoder",
        batch_size=int(cfg["encoder_training"].get("batch_size", 4)),
        distributed=False,
        rank=0,
        world_size=1,
        max_records=args.max_records,
    )
    loader = val_loader if args.split == "val" else test_loader
    if loader is None:
        raise RuntimeError(f"evaluation requires a {args.split} split")
    metrics = _evaluate_encoder(trainer, loader, args.max_val_batches)
    metrics.update(
        latent_dynamics(cfg, trainer, loader, args.max_val_batches)
    )
    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "metrics": metrics,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
