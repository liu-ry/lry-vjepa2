"""Compare latent predictors in a common raw-tactile target space."""

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

from app.vjepa_2_1.train_tactile_chain_stage_a import _resize_clip
from app.vjepa_2_1.train_tactile_chain_stage_b import (
    ChainStageB,
    _build_loaders,
    _encode_chains,
)


def _raw_future_summary(
    trainer: ChainStageB, batch: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = batch["tactile_chains"].to(trainer.device)
    if raw.dtype == torch.uint8:
        raw = raw.float().div_(255.0)
    else:
        raw = raw.float()
    batch_size, chains, time, channels, height, width = raw.shape
    future = raw[:, :, trainer.context : trainer.context + len(trainer.future_offsets)]
    future = _resize_clip(
        future.reshape(
            batch_size * chains,
            len(trainer.future_offsets),
            channels,
            height,
            width,
        ),
        trainer.image_size,
    )
    future = F.adaptive_avg_pool2d(
        future.reshape(-1, channels, *trainer.image_size), (7, 7)
    ).flatten(1)
    future = future.reshape(
        batch_size, chains, len(trainer.future_offsets), channels * 49
    )
    current = raw[:, :, trainer.context - 1]
    current = _resize_clip(
        current.reshape(batch_size * chains, 1, channels, height, width),
        trainer.image_size,
    ).squeeze(1)
    current = F.adaptive_avg_pool2d(current, (7, 7)).flatten(1)
    current = current.reshape(batch_size, chains, channels * 49)
    return future, current


def _predict(
    trainer: ChainStageB, batch: dict
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    clip, _ = trainer._prepare_tactile(batch)
    chain_valid = batch["chain_valid"].to(trainer.device, dtype=torch.bool)
    active_arm = torch.stack(
        (
            chain_valid[:, 0] | chain_valid[:, 1],
            chain_valid[:, 2] | chain_valid[:, 3],
        ),
        dim=1,
    )
    arm_valid = active_arm & batch["arm_valid"].to(
        trainer.device, dtype=torch.bool
    )
    with torch.no_grad(), torch.autocast(
        device_type="cuda",
        dtype=trainer.amp_dtype,
        enabled=trainer.use_amp,
    ):
        context_indices = list(range(trainer.context))
        tactile_context = _encode_chains(
            trainer.encoder,
            clip,
            context_indices,
            trainer.source_grid,
            trainer.latent_grid,
        )
        visual_context, _, _ = trainer._encode_visual(batch)
        output = trainer.predictor(
            tactile_context,
            context_indices,
            chain_valid,
            visual_context,
            list(range(visual_context.shape[2])),
            arm_valid,
            trainer._state_history(batch),
            trainer.context,
        )
    prediction = output["future_tactile"].mean(dim=-2)
    target, current = _raw_future_summary(trainer, batch)
    return prediction.float(), target.float(), current.float(), chain_valid


def _fit_ridge(
    train_x: list[torch.Tensor],
    train_y: list[torch.Tensor],
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.cat(train_x).float()
    y = torch.cat(train_y).float()
    x_mean = x.mean(dim=0, keepdim=True)
    x_std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_mean = y.mean(dim=0, keepdim=True)
    x = (x - x_mean) / x_std
    covariance = x.T @ x
    covariance.diagonal().add_(float(ridge) * len(x))
    coefficient = torch.linalg.solve(covariance, x.T @ (y - y_mean))
    return coefficient, x_mean, x_std, y_mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--predictor-checkpoint", required=True)
    parser.add_argument("--attention", default="per_frame_causal")
    parser.add_argument("--latent-grid", default="7x7")
    parser.add_argument("--train-batches", type=int, default=64)
    parser.add_argument("--test-batches", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    height, width = args.latent_grid.lower().split("x", 1)
    trainer = ChainStageB(
        cfg,
        "cuda",
        args.encoder_checkpoint,
        attention=args.attention,
        latent_grid=(int(height), int(width)),
    )
    state = torch.load(
        args.predictor_checkpoint, map_location="cuda", weights_only=False
    )
    for name in ("predictor", "contact_probe", "slip_probe"):
        getattr(trainer, name).load_state_dict(state[name])
    trainer.predictor.eval()
    trainer.visual.eval()
    train_loader, _, test_loader, split = _build_loaders(
        cfg,
        batch_size=int(cfg["predictor_training"]["batch_size"]),
        rank=0,
        world_size=1,
    )

    train_x: list[list[torch.Tensor]] = [[] for _ in range(4 * 30)]
    train_y: list[list[torch.Tensor]] = [[] for _ in range(4 * 30)]
    for batch_index, batch in enumerate(train_loader, start=1):
        prediction, target, _, valid = _predict(trainer, batch)
        for chain in range(4):
            for horizon in range(30):
                mask = valid[:, chain]
                if mask.any():
                    offset = chain * 30 + horizon
                    train_x[offset].append(
                        prediction[mask, chain, horizon].cpu()
                    )
                    train_y[offset].append(target[mask, chain, horizon].cpu())
        if args.train_batches and batch_index >= args.train_batches:
            break

    decoders = []
    for offset in range(4 * 30):
        decoders.append(
            _fit_ridge(train_x[offset], train_y[offset], args.ridge)
        )

    squared_error = torch.zeros(4, 30, dtype=torch.float64)
    copy_error = torch.zeros_like(squared_error)
    samples = torch.zeros(4, 30, dtype=torch.float64)
    with torch.no_grad():
        for batch_index, batch in enumerate(test_loader, start=1):
            prediction, target, current, valid = _predict(trainer, batch)
            for chain in range(4):
                mask = valid[:, chain]
                if not mask.any():
                    continue
                for horizon in range(30):
                    coefficient, x_mean, x_std, y_mean = decoders[
                        chain * 30 + horizon
                    ]
                    coefficient = coefficient.to(prediction.device)
                    x_mean = x_mean.to(prediction.device)
                    x_std = x_std.to(prediction.device)
                    y_mean = y_mean.to(prediction.device)
                    decoded = (
                        (prediction[mask, chain, horizon] - x_mean) / x_std
                    ) @ coefficient + y_mean
                    error = (decoded - target[mask, chain, horizon]).square().mean()
                    baseline = (
                        current[mask, chain] - target[mask, chain, horizon]
                    ).square().mean()
                    count = mask.sum().item()
                    squared_error[chain, horizon] += float(error) * count
                    copy_error[chain, horizon] += float(baseline) * count
                    samples[chain, horizon] += count
            if args.test_batches and batch_index >= args.test_batches:
                break

    mse = (squared_error / samples).tolist()
    copy = (copy_error / samples).tolist()
    report = {
        "split": split,
        "ridge": float(args.ridge),
        "train_batches": int(args.train_batches),
        "raw_future_mse_by_chain_horizon": mse,
        "raw_copy_mse_by_chain_horizon": copy,
        "raw_mse": float((squared_error / samples).mean()),
        "raw_copy": float((copy_error / samples).mean()),
        "raw_relative": float(
            (squared_error / samples).mean()
            / (copy_error / samples).mean().clamp_min(1e-12)
        ),
        "raw_mse_by_horizon": (squared_error / samples).mean(dim=0).tolist(),
        "raw_copy_by_horizon": (copy_error / samples).mean(dim=0).tolist(),
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "raw_mse": report["raw_mse"],
                "raw_copy": report["raw_copy"],
                "raw_relative": report["raw_relative"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
