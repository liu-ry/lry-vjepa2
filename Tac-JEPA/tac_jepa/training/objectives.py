"""Losses and metrics used unchanged by the tactile Stage A/B trainers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalized_context_latent_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    scale: float = 0.01,
    hand_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Absolute latent reconstruction loss for visible context tokens."""
    if predicted.shape != target.shape:
        raise ValueError(
            "context prediction and target shapes differ: "
            f"{tuple(predicted.shape)} vs {tuple(target.shape)}"
        )
    effective_scale = max(float(scale), 1e-6)
    error = F.smooth_l1_loss(
        predicted / effective_scale,
        target.detach() / effective_scale,
        reduction="none",
        beta=0.2,
    ).mean(dim=(-1, -2))
    copy_error = F.smooth_l1_loss(
        torch.zeros_like(target),
        target.detach() / effective_scale,
        reduction="none",
        beta=0.2,
    ).mean(dim=(-1, -2))
    if hand_valid is None:
        loss = error.mean()
        copy = copy_error.mean()
    else:
        if hand_valid.shape != predicted.shape[:1] + predicted.shape[2:3]:
            raise ValueError("hand_valid does not match hand dimension")
        mask = hand_valid[:, None, :].to(error.dtype)
        denominator = mask.sum().clamp_min(1.0)
        loss = (error * mask).sum() / denominator
        copy = (copy_error * mask).sum() / denominator
    cosine = F.cosine_similarity(
        predicted, target.detach(), dim=-1
    ).mean()
    return loss, copy, cosine

def _summary_4way(features: torch.Tensor) -> torch.Tensor:
    """Summarize ``[B,T,H,N,D]`` touch latents with four time/space statistics."""
    if features.ndim != 5:
        raise ValueError("summary features must be [B,T,H,N,D]")
    spatial_mean = features.mean(dim=-2)
    spatial_max = features.amax(dim=-2)
    return torch.stack(
        (
            spatial_mean.mean(dim=1),
            spatial_mean.amax(dim=1),
            spatial_max.mean(dim=1),
            spatial_max.amax(dim=1),
        ),
        dim=2,
    )

def _temporal_2way(features: torch.Tensor) -> torch.Tensor:
    """Summarize a ``[B,T,D]`` sequence with temporal mean and max."""
    if features.ndim != 3:
        raise ValueError("temporal summary features must be [B,T,D]")
    return torch.stack((features.mean(dim=1), features.amax(dim=1)), dim=1)

def _normalized_summary_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    hand_valid: torch.Tensor | None = None,
    hand_dim: int = 1,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"condition summary shape mismatch: {tuple(prediction.shape)} "
            f"!= {tuple(target.shape)}"
        )
    error = (
        F.normalize(prediction, dim=-1) - F.normalize(target, dim=-1)
    ).square().mean(dim=-1)
    if hand_valid is None:
        return error.mean()
    shape = [1] * error.ndim
    shape[0] = hand_valid.shape[0]
    shape[hand_dim] = hand_valid.shape[1]
    mask = hand_valid.view(*shape).expand_as(error).to(error.dtype)
    return (error * mask).sum() / mask.sum().clamp_min(1.0)

def _token_diversity_loss(tokens: torch.Tensor) -> torch.Tensor:
    """Penalize redundant condition slots and expose their cosine diversity."""
    if tokens.shape[1] < 2:
        return tokens.sum() * 0.0
    normalized = F.normalize(tokens, dim=-1)
    gram = normalized @ normalized.transpose(1, 2)
    eye = torch.eye(
        tokens.shape[1], device=tokens.device, dtype=torch.bool
    )[None]
    return gram.masked_select(~eye).square().mean()

def _dual_tactile_delta_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
    scale: float = 0.01,
    alpha: float = 1.0,
    direction_weight: float = 0.1,
    per_dim: bool = False,
    hand_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict future tactile change instead of nearly time-invariant tokens."""
    base = current.unsqueeze(1)
    predicted_delta = predicted - base
    target_delta = target - base
    magnitude = target_delta.square().mean(dim=-1).sqrt()
    if hand_valid is None:
        mask = torch.ones_like(magnitude)
        mean_magnitude = magnitude.mean(dim=(2, 3), keepdim=True)
    else:
        if hand_valid.shape != magnitude.shape[:1] + magnitude.shape[2:3]:
            raise ValueError("hand_valid does not match hand dimension")
        mask = hand_valid[:, None, :, None].expand_as(magnitude).to(magnitude.dtype)
        mean_magnitude = (
            magnitude * mask
        ).sum(dim=(2, 3), keepdim=True) / mask.sum(
            dim=(2, 3), keepdim=True
        ).clamp_min(1.0)
    weights = 1.0 + float(alpha) * (
        magnitude / mean_magnitude.clamp_min(1e-6)
    ).detach()
    weighted_mask = weights * mask
    scale = max(float(scale), 1e-6)
    if per_dim:
        global_scale = (
            target_delta.square().mean().sqrt().clamp_min(scale * 0.25)
        ).detach()
        dim_scale = (
            target_delta.square()
            .mean(dim=(0, 1, 2, 3), keepdim=True)
            .sqrt()
            .clamp_min(scale * 0.1)
            .detach()
        )
        effective_scale = torch.maximum(
            dim_scale, global_scale * 0.1
        )
    else:
        effective_scale = scale
    error = F.smooth_l1_loss(
        predicted_delta / effective_scale,
        target_delta / effective_scale,
        reduction="none",
        beta=0.2,
    ).mean(dim=-1)
    loss = (error * weighted_mask).sum() / weighted_mask.sum().clamp_min(1e-6)
    zero = F.smooth_l1_loss(
        torch.zeros_like(target_delta),
        target_delta / effective_scale,
        reduction="none",
        beta=0.2,
    ).mean(dim=-1)
    zero = (zero * weighted_mask).sum() / weighted_mask.sum().clamp_min(1e-6)
    direction_error = (
        F.normalize(predicted_delta, dim=-1)
        - F.normalize(target_delta, dim=-1)
    ).square().mean(dim=-1)
    copy_direction_error = (
        F.normalize(torch.zeros_like(target_delta), dim=-1)
        - F.normalize(target_delta, dim=-1)
    ).square().mean(dim=-1)
    direction_loss = (
        direction_error * weighted_mask
    ).sum() / weighted_mask.sum().clamp_min(1e-6)
    copy_direction = (
        copy_direction_error * weighted_mask
    ).sum() / weighted_mask.sum().clamp_min(1e-6)
    loss = loss + float(direction_weight) * direction_loss
    zero = zero + float(direction_weight) * copy_direction
    return loss, zero, target_delta.square().mean().sqrt()

def _auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.detach().float().reshape(-1)
    labels = labels.detach().float().reshape(-1) > 0.5
    positive = scores[labels]
    negative = scores[~labels]
    if positive.numel() == 0 or negative.numel() == 0:
        return 0.5
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    _, inverse, counts = torch.unique_consecutive(
        sorted_scores, return_inverse=True, return_counts=True
    )
    starts = counts.cumsum(dim=0) - counts
    average_ranks = (starts + 1 + starts + counts).to(scores.dtype) / 2
    ranks = average_ranks[inverse]
    positive_rank_sum = ranks[labels[order]].sum()
    n_positive = positive.numel()
    n_negative = negative.numel()
    statistic = positive_rank_sum - n_positive * (n_positive + 1) / 2
    return float(statistic / (n_positive * n_negative))
