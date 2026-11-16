"""Evaluate chain Stage-B latent prediction and four-query information coverage."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.vjepa_2_1.train_tactile_chain_stage_b import (
    ChainStageB,
    _build_loaders,
)


def _load_groups(path: str | None) -> set[str] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"group file must contain a JSON list: {path}")
    return {str(group) for group in payload}


def _filter_loader_groups(
    loader: DataLoader,
    groups: set[str] | None,
) -> DataLoader:
    if groups is None:
        return loader
    dataset = loader.dataset
    if dataset.dense_windows:
        raise ValueError(
            "group filtering expects a fixed-window dataset; "
            "set windows_per_episode before building evaluation loaders"
        )
    indices = []
    for item_index, item in enumerate(dataset.items):
        if str(item["group"]) not in groups:
            continue
        offset = item_index * dataset.windows_per_episode
        indices.extend(
            range(offset, offset + dataset.windows_per_episode)
        )
    if not indices:
        raise ValueError("group filtering produced an empty loader")
    kwargs = {
        "batch_size": loader.batch_size,
        "shuffle": False,
        "num_workers": loader.num_workers,
        "pin_memory": loader.pin_memory,
    }
    if loader.num_workers > 0:
        kwargs.update(
            prefetch_factor=loader.prefetch_factor,
            persistent_workers=loader.persistent_workers,
        )
    return DataLoader(Subset(dataset, indices), **kwargs)


def _collect(trainer, loader, max_batches: int):
    records = []
    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            trainer.step(batch, train=False)
            records.append(trainer._eval_features)
            if max_batches and index >= max_batches:
                break
    keys = records[0]
    return {
        key: torch.cat([record[key] for record in records], dim=0)
        for key in keys
    }


def _flatten_tactile(features: torch.Tensor, valid: torch.Tensor):
    batch, chains, dim = features.shape
    mask = valid.reshape(-1)
    return features.reshape(batch * chains, dim)[mask]


def _ridge_r2(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    ridge: float = 1e-2,
):
    train_x = train_x.float()
    test_x = test_x.float()
    train_y = train_y.float()
    test_y = test_y.float()
    x_mean = train_x.mean(dim=0, keepdim=True)
    x_std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_mean = train_y.mean(dim=0, keepdim=True)
    train_x = (train_x - x_mean) / x_std
    test_x = (test_x - x_mean) / x_std
    train_y = train_y - y_mean
    covariance = train_x.T @ train_x
    covariance.diagonal(dim1=0, dim2=1).add_(
        float(ridge) * len(train_x)
    )
    coefficients = torch.linalg.solve(
        covariance, train_x.T @ train_y
    )
    prediction = test_x @ coefficients + y_mean
    residual = (test_y - prediction).square().sum()
    total = (test_y - test_y.mean(dim=0, keepdim=True)).square().sum()
    return float(1.0 - residual / total.clamp_min(1e-8))


def _query_coverage(train_features, test_features):
    train_tokens = train_features["condition_tokens"]
    test_tokens = test_features["condition_tokens"]
    train_valid = train_features["chain_valid"].bool()
    test_valid = test_features["chain_valid"].bool()
    train_tokens = train_tokens[:, None].expand(
        -1, train_valid.shape[1], -1, -1
    )[train_valid]
    test_tokens = test_tokens[:, None].expand(
        -1, test_valid.shape[1], -1, -1
    )[test_valid]
    targets = {
        "history_tactile": (
            _flatten_tactile(
                train_features["history_latent"],
                train_features["chain_valid"],
            ),
            _flatten_tactile(
                test_features["history_latent"],
                test_features["chain_valid"],
            ),
        ),
        "future_tactile": (
            _flatten_tactile(
                train_features["future_latent"],
                train_features["chain_valid"],
            ),
            _flatten_tactile(
                test_features["future_latent"],
                test_features["chain_valid"],
            ),
        ),
    }
    result = {}
    for target_name, (train_y, test_y) in targets.items():
        result[target_name] = {}
        for token in range(train_tokens.shape[1]):
            result[target_name][f"token_{token}"] = _ridge_r2(
                train_tokens[:, token],
                train_y,
                test_tokens[:, token],
                test_y,
            )
        result[target_name]["all_4_tokens"] = _ridge_r2(
            train_tokens.flatten(1),
            train_y,
            test_tokens.flatten(1),
            test_y,
        )
        baseline = torch.zeros_like(test_y)
        residual = (test_y - baseline).square().sum()
        total = (test_y - test_y.mean(dim=0, keepdim=True)).square().sum()
        result[target_name]["zero_predictor_r2"] = float(
            1.0 - residual / total.clamp_min(1e-8)
        )
    return result


def _latent_metrics(features):
    prediction = features["predicted_future_latent"]
    target = features["future_latent"]
    valid = features["chain_valid"]
    output = {}
    for name, selector in (
        ("all", valid.any(dim=1)),
        ("single_arm", valid.sum(dim=1) == 2),
        ("dual_arm", valid.sum(dim=1) == 4),
    ):
        if not selector.any():
            continue
        pred = prediction[selector]
        tgt = target[selector]
        mask = valid[selector, :, None]
        normalized_mse = (
            F.normalize(pred, dim=-1) - F.normalize(tgt, dim=-1)
        ).square()
        output[f"{name}_normalized_mse"] = float(
            (normalized_mse * mask).sum() / mask.sum().clamp_min(1.0)
        )
        cosine = F.cosine_similarity(pred, tgt, dim=-1)
        output[f"{name}_cosine"] = float(
            (cosine * mask[..., 0]).sum() / mask[..., 0].sum().clamp_min(1.0)
        )
        output[f"{name}_samples"] = int(selector.sum())
    return output


def _query_geometry(features):
    tokens = features["condition_tokens"].float()
    normalized = F.normalize(tokens, dim=-1)
    pairwise = normalized @ normalized.transpose(1, 2)
    mask = ~torch.eye(tokens.shape[1], dtype=torch.bool)[None].expand_as(
        pairwise
    )
    centered = tokens.reshape(-1, tokens.shape[-1])
    centered = centered - centered.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    probability = energy / energy.sum().clamp_min(1e-12)
    effective_rank = torch.exp(
        -(probability * probability.clamp_min(1e-12).log()).sum()
    )
    return {
        "mean_offdiag_cosine": float(pairwise[mask].mean()),
        "effective_rank": float(effective_rank),
        "token_std": float(tokens.std(dim=1).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--predictor-checkpoint", required=True)
    parser.add_argument("--attention", required=True)
    parser.add_argument("--latent-grid", required=True)
    parser.add_argument("--predictor-depth", type=int, default=2)
    parser.add_argument("--condition-depth", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-batches", type=int, default=512)
    parser.add_argument("--test-batches", type=int, default=0)
    parser.add_argument(
        "--train-groups",
        default=None,
        help="Optional JSON list of episode groups for ridge fitting.",
    )
    parser.add_argument(
        "--test-groups",
        default=None,
        help="Optional JSON list of episode groups for held-out probes.",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    random.seed(731)
    torch.manual_seed(731)
    train_groups = _load_groups(args.train_groups)
    test_groups = _load_groups(args.test_groups)
    if train_groups is not None:
        cfg["windows_per_episode"] = 2
    cfg["predictor_training"]["predictor_depth"] = args.predictor_depth
    cfg["predictor_training"]["condition_depth"] = args.condition_depth
    height, width = args.latent_grid.lower().split("x", 1)
    trainer = ChainStageB(
        cfg,
        "cuda:0",
        args.encoder_checkpoint,
        attention=args.attention,
        latent_grid=(int(height), int(width)),
    )
    state = torch.load(
        args.predictor_checkpoint,
        map_location="cuda:0",
        weights_only=False,
    )
    trainer.predictor.load_state_dict(state["predictor"])
    trainer.contact_probe.load_state_dict(state["contact_probe"])
    trainer.slip_probe.load_state_dict(state["slip_probe"])
    trainer.predictor.eval()
    trainer.contact_probe.eval()
    trainer.slip_probe.eval()
    train_loader, _, test_loader, split = _build_loaders(
        cfg, batch_size=1, rank=0, world_size=1
    )
    train_loader = _filter_loader_groups(train_loader, train_groups)
    test_loader = _filter_loader_groups(test_loader, test_groups)
    generator = torch.Generator().manual_seed(731)
    train_order = torch.randperm(
        len(train_loader.dataset), generator=generator
    ).tolist()
    test_order = torch.randperm(
        len(test_loader.dataset), generator=generator
    ).tolist()
    loader_kwargs = {
        "batch_size": 1,
        "num_workers": int(cfg.get("num_workers", 4)),
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 1,
    }
    train_loader = DataLoader(
        train_loader.dataset,
        sampler=SubsetRandomSampler(train_order),
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_loader.dataset,
        sampler=SubsetRandomSampler(test_order),
        **loader_kwargs,
    )
    train_features = _collect(trainer, train_loader, args.train_batches)
    test_features = _collect(trainer, test_loader, args.test_batches)
    report = {
        "split": split,
        "latent_prediction": _latent_metrics(test_features),
        "query_coverage": _query_coverage(train_features, test_features),
        "query_geometry": _query_geometry(test_features),
        "n_train_samples": int(train_features["condition_tokens"].shape[0]),
        "n_test_samples": int(test_features["condition_tokens"].shape[0]),
        "group_filter": {
            "n_train_groups": (
                len(train_groups) if train_groups is not None else None
            ),
            "n_test_groups": (
                len(test_groups) if test_groups is not None else None
            ),
        },
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
