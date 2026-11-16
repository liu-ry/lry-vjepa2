"""Small tensor helpers required by the vendored visual backbone."""

from __future__ import annotations

import math

import torch


def apply_masks(x, masks, concat=True):
    all_x = []
    for mask in masks:
        keep = mask.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x.append(torch.gather(x, dim=1, index=keep))
    if not concat:
        return all_x
    return torch.cat(all_x, dim=0)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(value):
        return (1.0 + math.erf(value / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
