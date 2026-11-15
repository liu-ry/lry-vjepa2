"""Tests for the selected tactile encoder and dense world-model recipe."""

import torch

from app.vjepa_2_1.models.dense_tactile_predictor import DenseTactilePredictor
from app.vjepa_2_1.models.chain_world_predictor import ChainWorldPredictor
from app.vjepa_2_1.models.tactile_encoder import TactileEncoder
from app.vjepa_2_1.models.touch_latent_forecaster import (
    MaskedTouchLatentPredictor,
)
from app.vjepa_2_1.tactile_training_utils import (
    WarmupCosineMultiplier,
    _encode_tactile_indices,
    ensure_writable_output,
    resolve_device,
    unique_output_dir,
)
from app.vjepa_2_1.train_tactile_two_stage import (
    EncoderStage,
    TaskBalancedDistributedSampler,
    _auroc,
    _dense_tactile_relative,
    _dual_tactile_delta_loss,
    _normalized_summary_loss,
    _predictor_selection_loss,
    _summary_4way,
    _temporal_2way,
    _three_way_group_split,
)
from app.vjepa_2_1.train_tactile_chain_stage_b import (
    _align_temporal_embedding,
    _encode_chains,
    _encode_full_chains,
)
from app.vjepa_2_1.train_tactile_chain_stage_a import (
    ChainTactileEncoder,
)


def test_selected_encoder_has_294_tokens_per_hand_frame():
    encoder = TactileEncoder(
        embed_dim=32,
        in_channels=3,
        patch_size=16,
        depth=1,
        num_heads=4,
        img_size=(336, 224),
        use_rope=False,
    )
    assert encoder.max_spatial_tokens == 21 * 14 == 294
    output = encoder(torch.randn(1, 3, 336, 224))
    assert output.shape == (1, 1, 294, 32)


def test_stage_a_masks_only_tokens_inside_each_future_frame():
    stage = object.__new__(EncoderStage)
    stage.device = "cpu"
    stage.encoder = type("EncoderState", (), {"training": True})()
    stage.latent_mask_mode = "spatial_tokens"
    stage.section = {"latent_token_mask_ratio": 0.75}
    stage.cfg = {"seed": 0}
    mask = stage._stage_a_token_mask(
        batch=2,
        frames=6,
        context_frames=4,
        spatial_tokens=294,
    )
    assert mask.shape == (2, 6, 2, 294)
    assert not mask[:, :4].any()
    assert (~mask[:, 4:]).flatten(-2).any(dim=-1).all()
    mask_rate = mask[:, 4:].float().mean().item()
    assert 0.73 < mask_rate < 0.77


def test_tactile_image_encoder_tokens():
    encoder = TactileEncoder(
        embed_dim=32,
        in_channels=2,
        patch_size=4,
        depth=1,
        use_rope=False,
    )
    output = encoder(torch.randn(3, 5, 2, 16, 16))
    assert output.shape == (3, 5, 16, 32)


def test_tactile_encoder_accepts_fine_token_mask():
    encoder = TactileEncoder(
        embed_dim=32,
        in_channels=3,
        patch_size=4,
        depth=1,
        num_heads=4,
        img_size=(8, 8),
        use_rope=False,
    )
    mask = torch.zeros(2, 4, dtype=torch.bool)
    mask[:, 1] = True
    output = encoder(torch.randn(2, 3, 8, 8), token_mask=mask)
    assert output.shape == (2, 1, 4, 32)
    assert torch.isfinite(output).all()


def test_chain_tactile_encoder_prefix_is_causally_invariant():
    torch.manual_seed(0)
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
    clip = torch.randn(1, 4, 6, 32, 32)
    changed_future = clip.clone()
    changed_future[:, 2:] = torch.randn_like(changed_future[:, 2:])
    with torch.no_grad():
        original = encoder(clip)
        altered = encoder(changed_future)
    assert torch.allclose(original[:, :2], altered[:, :2], atol=1e-6)
    assert not torch.allclose(original[:, 2:], altered[:, 2:], atol=1e-6)


def test_tactile_vector_and_taxel_encoders():
    vector = TactileEncoder(embed_dim=32, input_dim=12, depth=1, use_rope=False)
    assert vector(torch.randn(3, 5, 12)).shape == (3, 5, 1, 32)
    taxel = TactileEncoder(
        embed_dim=32,
        patch_size=2,
        depth=1,
        use_rope=False,
        taxel_grid=(8, 8),
    )
    assert taxel(torch.randn(2, 4, 64)).shape == (2, 4, 16, 32)


def test_dense_predictor_emits_full_horizon_and_condition_tokens():
    model = DenseTactilePredictor(
        tactile_dim=32,
        visual_dim=24,
        state_dim=8,
        hidden_dim=32,
        depth=1,
        heads=4,
        max_time=35,
        num_hands=2,
        future_offsets=tuple(range(1, 31)),
        visual_future_offsets=tuple(range(2, 31, 2)),
        num_condition_tokens=4,
        drop_path=0.0,
    )
    output = model(
        torch.randn(2, 2, 2, 24),
        torch.randn(2, 4, 2, 2, 32),
        visual_indices=[1, 3],
        tactile_indices=[0, 1, 2, 3],
        context_frames=4,
        state_history=torch.randn(2, 4, 8),
    )
    assert output["future_tactile"].shape == (2, 30, 2, 1, 32)
    assert output["future_visual"].shape == (2, 15, 24)
    assert output["condition_tokens"].shape == (2, 4, 32)
    assert output["condition_future_tactile_trajectory"].shape == (
        2,
        30,
        2,
        32,
    )


def test_chain_predictor_mlp_projection_and_query_gradients():
    model = ChainWorldPredictor(
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
    tactile = torch.randn(2, 4, 4, 4, 16, requires_grad=True)
    output = model(
        tactile,
        [0, 1, 2, 3],
        torch.tensor([[True, True, False, True], [True, False, True, True]]),
        torch.randn(2, 2, 2, 12),
        [0, 1],
        torch.tensor([[True, True], [True, True]]),
        torch.randn(2, 2, 4, 8),
        4,
    )
    (output["future_tactile"].sum() + output["condition_tokens"].sum()).backward()
    assert tactile.grad is not None
    assert model.condition_query.grad is not None
    assert output["future_tactile"].shape == (2, 4, 3, 4, 16)
    assert "action" not in output
    assert not any("action" in name for name, _ in model.named_parameters())


def test_normed_projection_and_short_horizon_encoder_alignment():
    model = ChainWorldPredictor(
        tactile_dim=16,
        visual_dim=12,
        state_dim=8,
        spatial_tokens=4,
        hidden_dim=32,
        depth=1,
        condition_depth=1,
        heads=4,
        max_time=8,
        future_offsets=(1,),
        visual_future_offsets=(2,),
        num_condition_tokens=4,
        projection_type="normed_mlp",
        drop_path=0.0,
    )
    assert isinstance(model.tactile_in[0], torch.nn.LayerNorm)
    key = "temporal_adapter.time_embedding"
    source = {key: torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)}
    target = {key: torch.zeros(1, 34, 2)}
    aligned = _align_temporal_embedding(source, target)[key]
    assert aligned.shape == (1, 34, 2)
    assert torch.equal(aligned[:, :5], source[key])
    assert torch.equal(aligned[:, 5:], source[key][:, -1:].expand(-1, 29, -1))


def test_full_clip_target_preserves_causal_context_prefix():
    torch.manual_seed(1)
    encoder = ChainTactileEncoder(
        attention_mode="per_frame_causal",
        embed_dim=16,
        depth=1,
        heads=4,
        image_size=(32, 32),
        max_time=8,
        causal_depth=1,
        drop_path=0.0,
    ).eval()
    clip = torch.randn(2, 3, 8, 6, 32, 32)
    with torch.no_grad():
        full = _encode_full_chains(
            encoder, clip, (2, 2), (2, 2)
        )
        prefix = _encode_chains(
            encoder, clip, [0, 1, 2, 3], (2, 2), (2, 2)
        )
    assert full.shape == (2, 3, 8, 4, 16)
    assert torch.allclose(full[:, :, :4], prefix, atol=1e-6)


def test_dense_predictor_modalities_receive_gradients():
    model = DenseTactilePredictor(
        tactile_dim=32,
        visual_dim=24,
        state_dim=8,
        hidden_dim=32,
        depth=1,
        heads=4,
        max_time=35,
        num_hands=2,
        future_offsets=tuple(range(1, 31)),
        visual_future_offsets=tuple(range(2, 31, 2)),
        num_condition_tokens=4,
        drop_path=0.0,
    )
    with torch.no_grad():
        model.future_head[-1].weight.normal_(0.0, 0.02)
    visual = torch.randn(2, 2, 1, 24, requires_grad=True)
    state = torch.randn(2, 4, 8, requires_grad=True)
    output = model(
        visual,
        torch.randn(2, 4, 2, 2, 32),
        visual_indices=[1, 3],
        tactile_indices=[0, 1, 2, 3],
        context_frames=4,
        state_history=state,
    )
    output["future_tactile"].sum().backward()
    assert visual.grad is not None and visual.grad.abs().sum() > 0
    assert state.grad is not None and state.grad.abs().sum() > 0


def test_dense_predictor_masks_invalid_hand_outputs():
    model = DenseTactilePredictor(
        tactile_dim=32,
        visual_dim=24,
        state_dim=8,
        hidden_dim=32,
        depth=1,
        heads=4,
        max_time=8,
        num_hands=2,
        future_offsets=(1, 2),
        visual_future_offsets=(2,),
        num_condition_tokens=4,
        drop_path=0.0,
    )
    hand_valid = torch.tensor([[True, False], [False, True]])
    output = model(
        torch.randn(2, 2, 1, 24),
        torch.randn(2, 4, 2, 2, 32),
        visual_indices=[0, 1],
        tactile_indices=[0, 1, 2, 3],
        context_frames=4,
        state_history=torch.randn(2, 4, 8),
        hand_valid=hand_valid,
    )
    assert torch.count_nonzero(output["future_tactile"][0, :, 1]) == 0
    assert torch.count_nonzero(output["future_tactile"][1, :, 0]) == 0


def test_condition_summary_statistics():
    tactile = torch.randn(2, 7, 2, 3, 16)
    assert _summary_4way(tactile).shape == (2, 2, 4, 16)
    assert _temporal_2way(torch.randn(2, 5, 6)).shape == (2, 2, 6)


def test_masked_touch_predictor_emits_context_and_future_tokens():
    model = MaskedTouchLatentPredictor(
        tactile_dim=16,
        hidden_dim=16,
        depth=1,
        heads=4,
        max_time=8,
        num_hands=2,
        spatial_tokens=4,
        drop_path=0.0,
    )
    tokens = torch.randn(2, 4, 2, 4, 16, requires_grad=True)
    visible = torch.tensor([True, True, True, False]).view(1, 4, 1, 1)
    visible = visible.expand(2, 4, 2, 4)
    with torch.no_grad():
        model.context_head[-1].weight.normal_(0.0, 0.02)
        model.future_head[-1].weight.normal_(0.0, 0.02)
    output = model(
        tokens,
        memory_mask=visible,
        frame_indices=[0, 1, 2, 3],
        context_indices=[0, 1],
        future_indices=[2, 3],
        base_tokens=torch.randn(2, 2, 4, 16),
    )
    assert output["context_tactile"].shape == (2, 2, 2, 4, 16)
    assert output["future_tactile"].shape == (2, 2, 2, 4, 16)
    (output["context_tactile"].sum() + output["future_tactile"].sum()).backward()
    assert tokens.grad is not None and tokens.grad.abs().sum() > 0


def test_causal_tactile_delta_uses_requested_previous_frame():
    encoder = TactileEncoder(
        embed_dim=32,
        in_channels=6,
        patch_size=4,
        depth=1,
        use_rope=False,
    )
    tactile = torch.randn(2, 6, 3, 8, 8)
    encoded = _encode_tactile_indices(
        encoder, tactile, (1, 4), causal_delta=True
    )
    assert encoded.shape == (2, 2, 4, 32)


def test_dense_metrics_and_selection_use_all_future_offsets():
    relative = _dense_tactile_relative(0.2, 0.4, 4, 26)
    assert abs(relative - (4 * 0.2 + 26 * 0.4) / 30) < 1e-8
    section = {
        "selection_tactile_weight": 0.8,
        "selection_condition_weight": 0.2,
        "selection_visual_weight": 0.0,
        "selection_contact_weight": 0.0,
        "selection_slip_weight": 0.0,
    }
    score = _predictor_selection_loss(
        section,
        dense=True,
        tactile_relative=0.4,
        condition_loss=0.05,
        visual_relative=0.0,
        contact_auroc=0.5,
        slip_auroc=0.5,
    )
    assert abs(score - (0.8 * 0.4 + 0.2 * 0.05)) < 1e-8


def test_hand_masks_exclude_invalid_loss_values():
    hand_valid = torch.tensor([[True, False]])
    predicted = torch.zeros(1, 2, 2, 1, 8)
    current = torch.zeros(1, 2, 1, 8)
    target_a = torch.zeros(1, 2, 2, 1, 8)
    target_a[0, :, :, 0, 0] = torch.tensor([[1.0, 0.5], [0.2, 0.3]])
    target_b = target_a.clone()
    target_b[:, :, 1] = 100.0
    loss_a = _dual_tactile_delta_loss(
        predicted, target_a, current, hand_valid=hand_valid
    )[0]
    loss_b = _dual_tactile_delta_loss(
        predicted, target_b, current, hand_valid=hand_valid
    )[0]
    assert torch.isclose(loss_a, loss_b)
    summary_target = torch.zeros(1, 2, 4, 8)
    summary_target[:, 0] = 0.5
    summary_a = _normalized_summary_loss(
        torch.zeros(1, 2, 4, 8),
        summary_target,
        hand_valid=hand_valid,
    )
    target_summary_b = summary_target.clone()
    target_summary_b[:, 1] = 100.0
    summary_b = _normalized_summary_loss(
        torch.zeros(1, 2, 4, 8),
        target_summary_b,
        hand_valid=hand_valid,
    )
    assert torch.isclose(summary_a, summary_b)


def test_stable_group_split_is_independent_of_filtered_subset():
    names = [f"group-{index}" for index in range(100)]

    def items(values):
        return [{"group": name, "vision": f"{name}.mp4"} for name in values]

    train_a, val_a, test_a, _ = _three_way_group_split(
        items(names),
        val_ratio=0.15,
        test_ratio=0.1,
        seed=3,
        stable=True,
    )
    subset = [name for position, name in enumerate(names) if position % 3]
    train_b, val_b, test_b, _ = _three_way_group_split(
        items(subset),
        val_ratio=0.15,
        test_ratio=0.1,
        seed=3,
        stable=True,
    )
    partition_a = {
        "train": {names[index] for index in train_a},
        "val": {names[index] for index in val_a},
        "test": {names[index] for index in test_a},
    }
    partition_b = {
        "train": {subset[index] for index in train_b},
        "val": {subset[index] for index in val_b},
        "test": {subset[index] for index in test_b},
    }
    for split in ("train", "val", "test"):
        assert partition_a[split] & set(subset) == partition_b[split]


def test_task_balanced_sampler_and_auroc():
    class Dataset:
        windows_per_episode = 2

        def __len__(self):
            return 12

    tasks = ["common"] * 5 + ["rare"]
    sampler = TaskBalancedDistributedSampler(
        Dataset(),
        tasks,
        num_replicas=1,
        rank=0,
        seed=7,
        power=1.0,
    )
    first = list(sampler)
    assert first == list(sampler)
    assert sum(index >= 10 for index in first) >= 5
    scores = torch.tensor([0.1, 0.4, 0.4, 0.8, 0.2, 0.9])
    labels = torch.tensor([0, 0, 1, 1, 0, 1])
    positive = scores[labels.bool()]
    negative = scores[~labels.bool()]
    expected = (positive[:, None] > negative[None, :]).float()
    expected += 0.5 * (positive[:, None] == negative[None, :]).float()
    assert abs(_auroc(scores, labels) - float(expected.mean())) < 1e-6


def test_output_and_device_helpers(tmp_path=None):
    import tempfile
    from pathlib import Path

    root = Path(tempfile.mkdtemp()) if tmp_path is None else tmp_path
    base = root / "stage_a"
    assert unique_output_dir(base) == str(base)
    base.mkdir()
    assert Path(unique_output_dir(base)).name == "stage_a(1)"
    assert ensure_writable_output(
        "/tmp/tactile-run", ["/data/upload/recorder_1"]
    ) == "/tmp/tactile-run"
    assert resolve_device("cpu") == "cpu"


def test_warmup_cosine_keeps_parameter_group_ratios():
    optimizer = torch.optim.AdamW(
        [torch.nn.Parameter(torch.ones(1))],
        lr=1e-4,
    )
    optimizer.add_param_group(
        {"params": [torch.nn.Parameter(torch.ones(1))], "lr": 5e-5}
    )
    scheduler = WarmupCosineMultiplier(
        optimizer,
        total_steps=20,
        warmup_steps=4,
        min_lr_scale=0.1,
    )
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] > optimizer.param_groups[1]["lr"]
    ratio = (
        optimizer.param_groups[0]["lr"] / optimizer.param_groups[1]["lr"]
    )
    assert abs(ratio - 2.0) < 1e-6
