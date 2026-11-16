"""Stage A/B training entry points and shared helpers."""

from .objectives import (
    _auroc,
    _dual_tactile_delta_loss,
    _normalized_context_latent_loss,
    _normalized_summary_loss,
    _summary_4way,
    _temporal_2way,
    _token_diversity_loss,
)
from .splits import _three_way_group_split

__all__ = [
    "_auroc",
    "_dual_tactile_delta_loss",
    "_normalized_context_latent_loss",
    "_normalized_summary_loss",
    "_summary_4way",
    "_temporal_2way",
    "_three_way_group_split",
    "_token_diversity_loss",
]
