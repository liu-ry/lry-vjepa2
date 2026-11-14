"""Evaluate tactile/visual prediction and the four action-condition tokens.

The trained summary heads are useful optimization diagnostics, but they share
parameters with the predictor.  This script additionally fits independent
ridge probes on held-out condition tokens and evaluates them on the test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.vjepa_2_1.train_tactile_two_stage import (
    PredictorStage,
    _apply_overrides,
    _build_loaders,
    _load_config,
)


SUMMARY_PAIRS = {
    "history_tactile": (
        "condition_history_prediction",
        "condition_history_target",
    ),
    "future_tactile": (
        "condition_future_tactile_prediction",
        "condition_future_tactile_target",
    ),
    "history_visual": (
        "condition_history_visual_prediction",
        "condition_history_visual_target",
    ),
    "future_visual": (
        "condition_future_visual_prediction",
        "condition_future_visual_target",
    ),
    "history_state": (
        "condition_history_state_prediction",
        "condition_history_state_target",
    ),
}


def _load_checkpoint(trainer: PredictorStage, path: str, device: str) -> dict:
    state = torch.load(path, map_location=device, weights_only=False)
    trainer.predictor.load_state_dict(state["predictor"])
    trainer.contact_probe.load_state_dict(state["contact_probe"])
    trainer.slip_probe.load_state_dict(state["slip_probe"])
    if state.get("encoder") is not None:
        trainer.encoder.load_state_dict(state["encoder"])
    if trainer.fine_tune_encoder and state.get("encoder_target") is not None:
        trainer.encoder_target.load_state_dict(state["encoder_target"])
    return state


def _collect(
    trainer: PredictorStage,
    loader: DataLoader | None,
    max_batches: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float], int]:
    if loader is None:
        return {}, {}, 0
    trainer.predictor.eval()
    trainer.contact_probe.eval()
    trainer.slip_probe.eval()
    trainer.encoder.eval()
    collected: dict[str, list[torch.Tensor]] = {}
    metric_sums: dict[str, float] = {}
    batches = 0
    for batch in loader:
        with torch.inference_mode():
            metrics = trainer.step(batch, train=False)
        for key, value in metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
        for key, value in getattr(trainer, "_eval_predictions", {}).items():
            if key in (
                "condition_tokens",
                "condition_history_prediction",
                "condition_history_target",
                "condition_future_tactile_prediction",
                "condition_future_tactile_target",
                "condition_history_visual_prediction",
                "condition_history_visual_target",
                "condition_future_visual_prediction",
                "condition_future_visual_target",
                "condition_history_state_prediction",
                "condition_history_state_target",
                "visual_prediction",
                "visual_target",
                "visual_reference",
            ):
                collected.setdefault(key, []).append(value.float().cpu())
        batches += 1
        if max_batches and batches >= max_batches:
            break
    tensors = {
        key: torch.cat(values, dim=0)
        for key, values in collected.items()
    }
    averaged = {
        key: value / max(1, batches)
        for key, value in metric_sums.items()
    }
    return tensors, averaged, batches


def _ridge_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    alpha: float,
) -> dict[str, float]:
    mean = train_x.mean(dim=0, keepdim=True)
    scale = train_x.std(dim=0, keepdim=True).clamp_min(1e-5)
    train_norm = (train_x - mean) / scale
    test_norm = (test_x - mean) / scale
    train_norm = torch.cat(
        (train_norm, torch.ones(train_norm.shape[0], 1)), dim=1
    )
    test_norm = torch.cat(
        (test_norm, torch.ones(test_norm.shape[0], 1)), dim=1
    )
    gram = train_norm.T @ train_norm
    reg = torch.eye(gram.shape[0], dtype=gram.dtype) * float(alpha)
    reg[-1, -1] = 0.0
    weights = torch.linalg.solve(gram + reg, train_norm.T @ train_y)
    prediction = test_norm @ weights
    mse = F.mse_loss(prediction, test_y).item()
    target_energy = test_y.square().mean().clamp_min(1e-8)
    residual_energy = (test_y - prediction).square().mean()
    r2 = (1.0 - residual_energy / target_energy).item()
    cosine = F.cosine_similarity(
        prediction.flatten(1), test_y.flatten(1), dim=-1
    ).mean().item()
    return {
        "ridge_alpha": float(alpha),
        "test_mse": mse,
        "test_r2": r2,
        "test_cosine": cosine,
    }


def _summary_head_metrics(
    train: dict[str, torch.Tensor],
    test: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    result = {}
    for name, (pred_key, target_key) in SUMMARY_PAIRS.items():
        if pred_key not in train or target_key not in train:
            continue
        pred = F.normalize(train[pred_key].flatten(1), dim=-1)
        target = F.normalize(train[target_key].flatten(1), dim=-1)
        test_pred = F.normalize(test[pred_key].flatten(1), dim=-1)
        test_target = F.normalize(test[target_key].flatten(1), dim=-1)
        result[name] = {
            "validation_normalized_mse": F.mse_loss(pred, target).item(),
            "validation_cosine": F.cosine_similarity(
                pred, target, dim=-1
            ).mean().item(),
            "test_normalized_mse": F.mse_loss(
                test_pred, test_target
            ).item(),
            "test_cosine": F.cosine_similarity(
                test_pred, test_target, dim=-1
            ).mean().item(),
        }
    return result


def _token_metrics(
    train_tokens: torch.Tensor,
    test_tokens: torch.Tensor,
) -> dict[str, object]:
    def pairwise_cosine(tokens: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(tokens, dim=-1)
        gram = normalized @ normalized.transpose(1, 2)
        off_diagonal = gram[
            :, ~torch.eye(tokens.shape[1], dtype=torch.bool)
        ]
        return off_diagonal.mean(dim=-1)

    train_pairwise = pairwise_cosine(train_tokens)
    test_pairwise = pairwise_cosine(test_tokens)
    centered = train_tokens - train_tokens.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered.reshape(-1, centered.shape[-1]))
    probabilities = singular / singular.sum().clamp_min(1e-8)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    effective_rank = torch.exp(entropy).item()
    token_norms = train_tokens.norm(dim=-1)
    return {
        "num_tokens": int(train_tokens.shape[1]),
        "token_centroid_pairwise_cosine": F.cosine_similarity(
            train_tokens.mean(dim=0)[:, None],
            train_tokens.mean(dim=0)[None, :],
            dim=-1,
        ).tolist(),
        "validation_pairwise_cosine_mean": train_pairwise.mean().item(),
        "validation_pairwise_cosine_std": train_pairwise.std().item(),
        "test_pairwise_cosine_mean": test_pairwise.mean().item(),
        "test_pairwise_cosine_std": test_pairwise.std().item(),
        "token_norm_by_slot": token_norms.mean(dim=0).tolist(),
        "token_norm_std_by_slot": token_norms.std(dim=0).tolist(),
        "sample_centered_effective_rank": effective_rank,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-val-batches", type=int, default=64)
    parser.add_argument("--max-test-batches", type=int, default=64)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    cfg = _apply_overrides(_load_config(args.config), args.overrides)
    section = cfg["predictor_training"]
    trainer = PredictorStage(cfg, section, args.device, args.encoder_checkpoint)
    state = _load_checkpoint(trainer, args.checkpoint, args.device)
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
    val, val_metrics, val_batches = _collect(
        trainer, val_loader, args.max_val_batches
    )
    test, test_metrics, test_batches = _collect(
        trainer, test_loader, args.max_test_batches
    )
    if "condition_tokens" not in val or "condition_tokens" not in test:
        raise RuntimeError("checkpoint did not emit condition tokens")

    probes = {}
    train_x = val["condition_tokens"].flatten(1)
    test_x = test["condition_tokens"].flatten(1)
    for name, (_, target_key) in SUMMARY_PAIRS.items():
        if target_key not in val or target_key not in test:
            continue
        probes[name] = _ridge_probe(
            train_x,
            val[target_key].flatten(1),
            test_x,
            test[target_key].flatten(1),
            args.ridge_alpha,
        )

    visual = {}
    if "visual_prediction" in test:
        prediction = F.normalize(test["visual_prediction"], dim=-1)
        target = F.normalize(test["visual_target"], dim=-1)
        reference = F.normalize(test["visual_reference"], dim=-1)
        if reference.ndim == target.ndim - 1:
            reference = reference.unsqueeze(1)
        reference = reference.expand_as(target)
        prediction_error = (prediction - target).square()
        copy_error = (reference - target).square()
        prediction_delta = prediction - reference
        target_delta = target - reference
        delta_scale = target_delta.square().mean().sqrt().clamp_min(1e-6)
        delta_error = ((prediction_delta - target_delta) / delta_scale).square()
        delta_copy_error = (target_delta / delta_scale).square()
        visual = {
            "test_cosine": F.cosine_similarity(
                prediction, target, dim=-1
            ).mean().item(),
            "test_copy_last_cosine": F.cosine_similarity(
                reference, target, dim=-1
            ).mean().item(),
            "test_normalized_mse": prediction_error.mean().item(),
            "test_copy_last_normalized_mse": copy_error.mean().item(),
            "test_relative_improvement": 1.0
            - prediction_error.mean().item()
            / max(copy_error.mean().item(), 1e-8),
            "test_delta_mse": delta_error.mean().item(),
            "test_delta_copy_mse": delta_copy_error.mean().item(),
            "test_delta_relative_improvement": 1.0
            - delta_error.mean().item()
            / max(delta_copy_error.mean().item(), 1e-8),
            "by_horizon": {
                str(offset): {
                    "cosine": F.cosine_similarity(
                        prediction[:, index], target[:, index], dim=-1
                    ).mean().item(),
                    "copy_last_cosine": F.cosine_similarity(
                        reference[:, index], target[:, index], dim=-1
                    ).mean().item(),
                    "delta_cosine": F.cosine_similarity(
                        prediction_delta[:, index],
                        target_delta[:, index],
                        dim=-1,
                    ).mean().item(),
                }
                for index, offset in enumerate(
                    cfg["visual_future_offsets"]
                )
            },
        }

    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "step": int(state.get("step", -1)),
        "batches": {
            "validation": val_batches,
            "test": test_batches,
        },
        "samples": {
            "validation": int(val["condition_tokens"].shape[0]),
            "test": int(test["condition_tokens"].shape[0]),
        },
        "split": {
            key: split[key]
            for key in (
                "n_train_windows",
                "n_val_windows",
                "n_test_windows",
            )
        },
        "trained_summary_heads": _summary_head_metrics(val, test),
        "independent_ridge_probes": probes,
        "condition_tokens": _token_metrics(
            val["condition_tokens"], test["condition_tokens"]
        ),
        "visual_prediction": visual,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
