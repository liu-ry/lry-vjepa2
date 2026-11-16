"""Fast contract tests that do not require datasets or checkpoints."""

from __future__ import annotations

import torch

from tac_jepa.models.chain_encoder import (
    ChainTactileEncoder,
    _causal_delta_clip,
    _encode_chains,
)
from tac_jepa.models.predictor import ChainWorldPredictor


def test_encoder_and_predictor_contract():
    torch.manual_seed(0)
    encoder = ChainTactileEncoder(
        attention_mode="per_frame_causal",
        embed_dim=16,
        depth=1,
        heads=4,
        image_size=(32, 32),
        max_time=4,
        causal_depth=1,
        drop_path=0.0,
    ).eval()
    raw_clip = torch.randn(2, 4, 4, 3, 32, 32)
    clip = _causal_delta_clip(
        raw_clip.reshape(2 * 4, 4, 3, 32, 32)
    ).reshape(2, 4, 4, 6, 32, 32)
    tactile_context = _encode_chains(
        encoder,
        clip,
        [0, 1, 2, 3],
        source_grid=(2, 2),
        target_grid=(2, 2),
    )
    assert tactile_context.shape == (2, 4, 4, 4, 16)

    predictor = ChainWorldPredictor(
        tactile_dim=16,
        visual_dim=12,
        state_dim=8,
        spatial_tokens=4,
        hidden_dim=32,
        depth=1,
        condition_depth=1,
        heads=4,
        max_time=8,
        future_offsets=(1, 2, 3),
        visual_future_offsets=(2,),
        num_condition_tokens=4,
        projection_type="mlp",
        drop_path=0.0,
    )
    output = predictor(
        tactile_context,
        [0, 1, 2, 3],
        torch.ones(2, 4, dtype=torch.bool),
        torch.randn(2, 2, 2, 12),
        [0, 1],
        torch.ones(2, 2, dtype=torch.bool),
        torch.randn(2, 2, 4, 8),
        4,
    )
    assert output["future_tactile"].shape == (2, 4, 3, 4, 16)
    assert output["future_visual"].shape == (2, 2, 1, 12)
    assert output["condition_tokens"].shape == (2, 4, 32)


def test_causal_prefix_is_invariant():
    torch.manual_seed(1)
    encoder = ChainTactileEncoder(
        attention_mode="per_frame_causal",
        embed_dim=16,
        depth=1,
        heads=4,
        image_size=(32, 32),
        max_time=6,
        causal_depth=1,
        drop_path=0.0,
    ).eval()
    clip = torch.randn(1, 6, 6, 32, 32)
    changed = clip.clone()
    changed[:, 2:] = torch.randn_like(changed[:, 2:])
    with torch.no_grad():
        original = encoder(clip)
        altered = encoder(changed)
    assert torch.allclose(original[:, :2], altered[:, :2], atol=1e-6)
