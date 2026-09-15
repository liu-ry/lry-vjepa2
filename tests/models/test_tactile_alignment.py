import torch
import torch.nn as nn

from app.vjepa_2_1.models.multimodal_predictor import FutureLatentPredictor
from app.vjepa_2_1.models.residual_correction import LatentResidualController
from app.vjepa_2_1.models.tactile_alignment import TactileAlignment, TactileEncoder
from app.vjepa_2_1.train_tactile_alignment import (
    MultimodalTrainer,
    WarmupCosineMultiplier,
    best_checkpoint,
    future_offsets_from_config,
    output_from_config,
    prune_checkpoints,
    record_checkpoint_metrics,
    resume_from_config,
    save_tensorboard_curves,
    should_save_epoch,
    split_indices,
    unique_output_dir,
    window_from_config,
    sample_future_offsets,
)
from app.vjepa_2_1.visualize_latents import (
    copy_last,
    cosine_maps,
    factor_hw,
    pca_basis,
    pca_maps,
    step_cosine,
)


def test_vjepa_predictor_accepts_noncanonical_depth():
    from app.vjepa_2_1.models.predictor import vit_predictor

    model = vit_predictor(
        embed_dim=32, predictor_embed_dim=16, depth=1, num_heads=2, use_rope=False
    )
    assert model.hierarchical_layers == [0]


def test_unique_output_dir_appends_incrementing_suffix():
    import tempfile
    from pathlib import Path

    root = Path(tempfile.mkdtemp())
    base = root / "tactile_align"
    assert unique_output_dir(base) == str(base)
    base.mkdir()
    first = Path(unique_output_dir(base))
    assert first.name == "tactile_align(1)"
    first.mkdir()
    second = Path(unique_output_dir(base))
    assert second.name == "tactile_align(2)"
    second.mkdir()
    (root / "tactile_align(5)").mkdir()
    assert Path(unique_output_dir(base)).name == "tactile_align(6)"
    (base / "checkpoint_0001.pt").write_bytes(b"a")
    assert unique_output_dir(base, resume=str(base / "checkpoint_0001.pt")) == str(base)
    (first / "checkpoint_0002.pt").write_bytes(b"b")
    assert unique_output_dir(base, resume=str(first / "checkpoint_0002.pt")) == str(first)


def test_save_tensorboard_curves_writes_pngs():
    import tempfile
    from pathlib import Path
    from torch.utils.tensorboard import SummaryWriter

    root = Path(tempfile.mkdtemp())
    log_dir = root / "tensorboard"
    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_scalar("loss/train", 1.0, 1)
    writer.add_scalar("loss/val", 1.2, 1)
    writer.add_scalar("loss/train", 0.8, 2)
    writer.add_scalar("loss/val", 0.9, 2)
    writer.add_scalar("loss/train_step", 1.1, 1)
    writer.add_scalar("loss_global/train", 0.7, 1)
    writer.flush()
    writer.close()
    saved = save_tensorboard_curves(log_dir, root / "curves")
    names = {path.name for path in saved}
    assert "loss.png" in names
    assert "loss_train_step.png" in names
    assert "loss_global.png" in names
    assert "overview.png" in names
    for path in saved:
        assert path.is_file() and path.stat().st_size > 0


def test_should_save_epoch_every_ten_and_last():
    saved = [i for i in range(1, 51) if should_save_epoch(i, 10, is_last=(i == 50))]
    assert saved == [10, 20, 30, 40, 50]
    assert should_save_epoch(7, 10, is_last=True)
    assert not should_save_epoch(7, 10, is_last=False)


def test_output_from_config_picks_stage_directory():
    cfg = {"training": {"output": {"align": "out/align", "joint": "out/joint"}}}
    assert output_from_config(cfg, "align") == "out/align"
    assert output_from_config(cfg, "joint") == "out/joint"
    assert output_from_config(cfg, "joint", override="tmp/joint") == "tmp/joint"
    try:
        output_from_config({"training": {"output": {"align": "out/align"}}}, "joint")
    except ValueError as exc:
        assert "output.joint" in str(exc)
        return
    raise AssertionError("expected ValueError when the stage output directory is missing")


def test_best_checkpoint_uses_lowest_recorded_loss(tmp_path=None):
    import tempfile
    from pathlib import Path

    root = Path(tempfile.mkdtemp()) if tmp_path is None else Path(tmp_path)
    (root / "checkpoint_0001.pt").write_bytes(b"a")
    (root / "checkpoint_0002.pt").write_bytes(b"b")
    (root / "checkpoint_step_0003.pt").write_bytes(b"c")
    record_checkpoint_metrics(root, "checkpoint_0001.pt", 1, {"loss": 0.40})
    record_checkpoint_metrics(root, "checkpoint_0002.pt", 2, {"loss": 0.55})
    chosen = best_checkpoint(root)
    assert chosen.name == "checkpoint_0001.pt"


def test_best_checkpoint_falls_back_to_latest_epoch():
    import tempfile
    from pathlib import Path

    root = Path(tempfile.mkdtemp())
    (root / "checkpoint_step_0009.pt").write_bytes(b"s")
    (root / "checkpoint_0001.pt").write_bytes(b"a")
    (root / "checkpoint_0003.pt").write_bytes(b"b")
    assert best_checkpoint(root).name == "checkpoint_0003.pt"


def test_joint_resume_defaults_to_best_align_checkpoint():
    import tempfile
    from pathlib import Path

    root = Path(tempfile.mkdtemp())
    align = root / "align"
    align.mkdir()
    (align / "checkpoint_0001.pt").write_bytes(b"a")
    (align / "checkpoint_0002.pt").write_bytes(b"b")
    record_checkpoint_metrics(align, "checkpoint_0001.pt", 1, {"loss": 0.2})
    record_checkpoint_metrics(align, "checkpoint_0002.pt", 2, {"loss": 0.9})
    cfg = {"training": {"output": {"align": str(align), "joint": str(root / "joint")}}}
    assert resume_from_config(cfg, "align") is None
    assert Path(resume_from_config(cfg, "joint")).name == "checkpoint_0001.pt"
    assert resume_from_config(cfg, "joint", override="explicit.pt") == "explicit.pt"
    assert resume_from_config(cfg, "joint", override=False) is None


def test_prune_checkpoints_keeps_best_and_latest():
    import tempfile
    from pathlib import Path

    root = Path(tempfile.mkdtemp())
    for i in range(1, 5):
        (root / f"checkpoint_{i:04d}.pt").write_bytes(b"x" * 10)
    (root / "checkpoint_step_0009.pt").write_bytes(b"s")
    record_checkpoint_metrics(root, "checkpoint_0002.pt", 2, {"loss": 0.1})
    record_checkpoint_metrics(root, "checkpoint_0004.pt", 4, {"loss": 0.5})
    removed = prune_checkpoints(root, keep_last=1)
    names = sorted(p.name for p in root.glob("*.pt"))
    assert "checkpoint_0002.pt" in names
    assert "checkpoint_0004.pt" in names
    assert "checkpoint_0001.pt" not in names
    assert "checkpoint_step_0009.pt" not in names
    assert "checkpoint_0001.pt" in removed


def test_split_indices_is_disjoint_and_reproducible():
    train_a, val_a = split_indices(10, val_ratio=0.2, seed=0)
    train_b, val_b = split_indices(10, val_ratio=0.2, seed=0)
    train_c, val_c = split_indices(10, val_ratio=0.2, seed=1)
    assert len(val_a) == 2
    assert len(train_a) == 8
    assert sorted(train_a + val_a) == list(range(10))
    assert set(train_a).isdisjoint(val_a)
    assert (train_a, val_a) == (train_b, val_b)
    assert (train_a, val_a) != (train_c, val_c)
    assert split_indices(10, val_ratio=0.0, seed=0)[1] == []


def test_window_from_config_n_and_m():
    context, horizon, tubelet = window_from_config(
        {"context": 4, "horizon": 30, "model": {"tubelet_size": 1}}
    )
    assert (context, horizon, tubelet) == (4, 30, 1)
    context, horizon, tubelet = window_from_config(
        {"input_frames": 4, "output_frames": 16, "model": {"tubelet_size": 2}}
    )
    assert (context, horizon, tubelet) == (4, 16, 2)
    try:
        window_from_config({"context": 3, "horizon": 8, "model": {"tubelet_size": 2}})
    except ValueError as exc:
        assert "multiples" in str(exc)
        return
    raise AssertionError("expected ValueError when n/m are not multiples of tubelet")


def test_sample_future_offsets_is_sorted_and_on_grid():
    offsets = sample_future_offsets(30, 4, tubelet=1)
    assert len(offsets) == 4
    assert offsets == tuple(sorted(set(offsets)))
    assert all(1 <= x <= 30 for x in offsets)


def test_default_future_offsets_follow_tubelet_grid():
    assert future_offsets_from_config({}, tubelet=2, horizon=8) == (2, 4, 6, 8)


def test_tactile_image_encoder_tokens():
    enc = TactileEncoder(embed_dim=32, in_channels=2, patch_size=4, depth=1, use_rope=False)
    y = enc(torch.randn(3, 5, 2, 16, 16))
    assert y.shape == (3, 5, 16, 32)


def test_tactile_image_encoder_tubelets():
    enc = TactileEncoder(
        embed_dim=32, in_channels=2, patch_size=4, tubelet_size=2, depth=1, use_rope=False
    )
    y = enc(torch.randn(2, 6, 2, 16, 16))
    assert y.shape == (2, 3, 16, 32)


def test_tactile_vector_encoder_tokens():
    enc = TactileEncoder(embed_dim=32, input_dim=12, depth=1, use_rope=False)
    y = enc(torch.randn(3, 5, 12))
    assert y.shape == (3, 5, 1, 32)


def test_tactile_taxel_grid_tokens():
    enc = TactileEncoder(
        embed_dim=32, patch_size=2, depth=1, use_rope=False, taxel_grid=(8, 8)
    )
    y = enc(torch.randn(2, 4, 64))
    assert y.shape == (2, 4, 16, 32)


def test_alignment_losses_backpropagate_only_to_tactile():
    tactile = torch.randn(4, 3, 2, 16, requires_grad=True)
    visual = torch.randn(4, 3, 2, 16, requires_grad=True)
    module = TactileAlignment(16, 16, projection_dim=8)
    loss, metrics = module(tactile, visual)
    loss.backward()
    assert torch.isfinite(loss)
    assert tactile.grad is not None
    assert visual.grad is None
    assert set(metrics) == {"loss_global", "loss_latent", "loss_temporal"}


def test_predictor_rollout_four_to_many():
    pred = FutureLatentPredictor(
        visual_dim=8, tactile_dim=8, hidden_dim=16, depth=1, heads=2,
        horizon=2, max_context=8, max_visual_tokens=4, max_tactile_tokens=4,
    )
    visual = torch.randn(1, 4, 4, 8)
    tactile = torch.randn(1, 4, 4, 8)
    pv, ph = pred.rollout(visual, tactile, steps=50, window=4)
    assert pv.shape == (1, 50, 4, 8)
    assert ph.shape == (1, 50, 4, 8)


def test_predictor_preserves_spatial_tokens():
    pred = FutureLatentPredictor(
        visual_dim=8, tactile_dim=12, hidden_dim=16, depth=1, heads=2,
        horizon=2, max_context=8, max_visual_tokens=4, max_tactile_tokens=3,
    )
    visual = torch.randn(2, 3, 4, 8)
    tactile = torch.randn(2, 3, 3, 12)
    pv, ph, cond = pred(visual, tactile)
    assert pv.shape == (2, 2, 4, 8)
    assert ph.shape == (2, 2, 3, 12)
    assert cond.shape == (2, 16, 16)


def test_predictor_accepts_runtime_sparse_offsets():
    pred = FutureLatentPredictor(
        visual_dim=8, tactile_dim=8, hidden_dim=16, depth=1, heads=2,
        horizon=6, max_context=4, max_visual_tokens=4, max_tactile_tokens=3,
        future_offsets=(5, 10, 15, 20, 25, 30),
    )
    visual = torch.randn(1, 4, 4, 8)
    tactile = torch.randn(1, 4, 3, 8)
    pv, ph, cond = pred(visual, tactile, future_offsets=(3, 6, 10, 20))
    assert pv.shape == (1, 4, 4, 8)
    assert ph.shape == (1, 4, 3, 8)
    assert cond.shape == (1, 16, 16)


def test_predictor_accepts_sparse_configured_offsets():
    # 配置的 horizon 是连续 rollout 容量；稀疏 waypoint 计划可以更少 query。
    pred = FutureLatentPredictor(
        visual_dim=8, tactile_dim=8, hidden_dim=16, depth=1, heads=2,
        horizon=30, max_context=4, max_visual_tokens=4, max_tactile_tokens=3,
        future_offsets=(5, 10, 15, 20, 25, 30),
    )
    visual = torch.randn(1, 4, 4, 8)
    tactile = torch.randn(1, 4, 3, 8)
    pv, ph, cond = pred(visual, tactile)
    assert pv.shape == (1, 6, 4, 8)
    assert ph.shape == (1, 6, 3, 8)
    assert cond.shape == (1, 16, 16)


def test_predictor_sparse_plan_can_query_capacity_not_in_initial_waypoints():
    pred = FutureLatentPredictor(
        visual_dim=8, tactile_dim=8, hidden_dim=16, depth=1, heads=2,
        horizon=30, max_context=4, max_visual_tokens=4, max_tactile_tokens=3,
        future_offsets=(5, 10), max_future_offset=30,
    )
    visual = torch.randn(1, 4, 4, 8)
    tactile = torch.randn(1, 4, 3, 8)
    pv, ph, cond = pred(visual, tactile, future_offsets=(20, 30))
    assert pv.shape == (1, 2, 4, 8)
    assert ph.shape == (1, 2, 3, 8)
    assert cond.shape == (1, 16, 16)


def test_residual_controller_selects_waypoint_and_preserves_suffix_shape():
    controller = LatentResidualController(latent_dim=8, action_dim=7, hidden_dim=16)
    predicted = torch.randn(2, 3, 4, 8)
    observed = torch.randn(2, 4, 8)
    actions = torch.randn(2, 5, 7)
    corrected = controller.correct_at(predicted, observed, actions, 10, (5, 10, 15))
    assert corrected.shape == actions.shape

    # 选出的 [B, N, D] 是一张空间 token 格子，不是 N 个时间 waypoint。
    corrected_single = controller.correct_at(predicted[:, :1], observed[:, 0], actions, 5, (5,))
    assert corrected_single.shape == actions.shape


def test_residual_mlp_does_not_use_remaining_actions():
    controller = LatentResidualController(latent_dim=8, action_dim=7, hidden_dim=16)
    predicted = torch.randn(2, 8)
    observed = torch.randn(2, 8)
    actions_a = torch.randn(2, 5, 7)
    actions_b = torch.randn(2, 5, 7)
    residual_a = controller(predicted, observed, actions_a) - actions_a
    residual_b = controller(predicted, observed, actions_b) - actions_b
    assert torch.allclose(residual_a, residual_b)


def test_resnet_backbone_is_rejected():
    try:
        TactileEncoder(embed_dim=32, backbone="resnet18", depth=1)
    except ValueError as exc:
        assert "ResNet" in str(exc)
        return
    raise AssertionError("expected ValueError for ResNet backbones")


class _DummyVisual(nn.Module):
    def __init__(self, dim=16, tokens=8):
        super().__init__()
        self.embed_dim = dim
        self.tokens = tokens
        self.proj = nn.Linear(3, dim)

    def forward(self, x, training=False):
        batch, _, time, _, _ = x.shape
        pooled = x.mean(dim=(3, 4)).permute(0, 2, 1)
        z = self.proj(pooled).unsqueeze(2).expand(batch, time, self.tokens, self.embed_dim)
        return z.reshape(batch, time * self.tokens, self.embed_dim)


def test_warmup_cosine_peaks_then_decays_and_keeps_group_ratios():
    a = torch.nn.Parameter(torch.zeros(1))
    b = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([{"params": [a], "lr": 0.1}, {"params": [b], "lr": 0.3}])
    sched = WarmupCosineMultiplier(
        opt, total_steps=10, warmup_steps=2, start_lr_scale=0.0, min_lr_scale=0.0
    )
    lrs = [sched.get_last_lr()]
    for _ in range(10):
        sched.step()
        lrs.append(sched.get_last_lr())
    assert abs(lrs[0][0] - 0.05) < 1e-6
    assert abs(lrs[0][1] - 0.15) < 1e-6
    assert abs(lrs[1][0] - 0.1) < 1e-6
    assert abs(lrs[1][1] - 0.3) < 1e-6
    peak = max(lr[0] for lr in lrs)
    assert abs(peak - 0.1) < 1e-6
    assert lrs[-1][0] < lrs[2][0]
    assert abs(lrs[-1][0]) < 1e-6
    assert all(abs(lr[1] / lr[0] - 3.0) < 1e-6 for lr in lrs if lr[0] > 0)


def test_trainer_joint_step_predicts_spatial_future():
    visual = _DummyVisual()
    tactile = TactileEncoder(embed_dim=16, in_channels=2, patch_size=4, depth=1, use_rope=False)
    predictor = FutureLatentPredictor(
        visual_dim=16, tactile_dim=16, hidden_dim=16, depth=1, heads=2,
        horizon=2, max_visual_tokens=8, max_tactile_tokens=16,
    )
    trainer = MultimodalTrainer(
        visual, tactile, predictor, device="cpu", freeze_visual=True, visual_tubelet_size=1
    )
    batch = {
        "vision": torch.randn(2, 6, 3, 16, 16),
        "tactile": torch.randn(2, 6, 2, 16, 16),
    }
    metrics = trainer.step(batch, context=4, horizon=2, stage="joint")
    assert torch.isfinite(metrics["loss"])
    assert "loss_future_visual" in metrics and "loss_future_tactile" in metrics


def test_predictor_future_loss_is_token_level():
    pred = FutureLatentPredictor(
        visual_dim=8, tactile_dim=8, hidden_dim=16, depth=1, heads=2, horizon=1,
        max_visual_tokens=4, max_tactile_tokens=4,
    )
    visual = torch.randn(2, 2, 4, 8, requires_grad=True)
    tactile = torch.randn(2, 2, 4, 8, requires_grad=True)
    pv, ph, _ = pred(visual.detach(), tactile)
    target_v = torch.randn(2, 1, 4, 8)
    target_h = torch.randn(2, 1, 4, 8)
    loss = torch.nn.functional.mse_loss(pv, target_v) + torch.nn.functional.mse_loss(ph, target_h)
    loss.backward()
    assert tactile.grad is not None
    assert visual.grad is None
    assert pv.shape[-2] == visual.shape[-2]


def test_factor_hw_and_pca_maps():
    assert factor_hw(256) == (16, 16)
    assert factor_hw(196) == (14, 14)
    tokens = torch.randn(4, 16, 8)
    mean, pcs = pca_basis(tokens, k=3)
    rgb = pca_maps(tokens, mean, pcs, hw=(4, 4))
    assert rgb.shape == (4, 4, 4, 3)
    assert rgb.min() >= 0 and rgb.max() <= 1
    pred = tokens[2:]
    target = tokens[2:] + 0.01
    err = cosine_maps(pred, target, (4, 4))
    assert err.shape == (2, 4, 4)
    copied = copy_last(tokens[:2], horizon=2)
    assert copied.shape == (2, 16, 8)
    assert torch.allclose(copied[0], tokens[1])
    cos = step_cosine(tokens[2:], tokens[2:])
    assert len(cos) == 2
    assert min(cos) > 0.99


def _tiny_predictor(**kwargs):
    defaults = dict(
        visual_dim=8, tactile_dim=8, hidden_dim=16, depth=1, heads=2,
        horizon=2, max_context=4, max_visual_tokens=4, max_tactile_tokens=3,
        num_condition_tokens=4, condition_out_dim=12,
    )
    defaults.update(kwargs)
    return FutureLatentPredictor(**defaults)


def test_predictor_condition_tokens_shape():
    pred = _tiny_predictor()
    pv, ph, cond = pred(torch.randn(2, 3, 4, 8), torch.randn(2, 3, 3, 8))
    assert pv.shape == (2, 2, 4, 8)
    assert ph.shape == (2, 2, 3, 8)
    assert cond.shape == (2, 4, 12)


def test_predictor_zero_condition_tokens_returns_none():
    pred = _tiny_predictor(num_condition_tokens=0)
    pv, ph, cond = pred(torch.randn(1, 3, 4, 8), torch.randn(1, 3, 3, 8))
    assert pv.shape == (1, 2, 4, 8)
    assert ph.shape == (1, 2, 3, 8)
    assert cond is None


def test_attention_mask_is_prefix_encoder_decoder():
    pred = _tiny_predictor()
    context_len, future_len, cond_len = 5, 7, 4
    mask = pred._attention_mask(
        context_len, future_len, cond_len, torch.float32, torch.device("cpu")
    )
    blocked = torch.finfo(torch.float32).min
    prefix = context_len + future_len
    assert mask.shape == (prefix + cond_len, prefix + cond_len)
    assert torch.all(mask[:context_len, :context_len] == 0)
    assert torch.all(mask[:context_len, context_len:] == blocked)
    assert torch.all(mask[context_len:prefix, :prefix] == 0)
    assert torch.all(mask[context_len:prefix, prefix:] == blocked)
    assert torch.all(mask[prefix:] == 0)


def test_context_mask_applies_without_condition_queries():
    pred = _tiny_predictor(num_condition_tokens=0)
    mask = pred._attention_mask(4, 6, 0, torch.float32, torch.device("cpu"))
    blocked = torch.finfo(torch.float32).min
    assert torch.all(mask[:4, 4:] == blocked)
    assert torch.all(mask[4:, :] == 0)


def test_condition_queries_do_not_change_future_latents():
    pred = _tiny_predictor()
    pred.eval()
    visual = torch.randn(1, 3, 4, 8)
    tactile = torch.randn(1, 3, 3, 8)
    pv1, ph1, cond1 = pred(visual, tactile)
    pred.condition_query.data.add_(1.0)
    pv2, ph2, cond2 = pred(visual, tactile)
    assert torch.allclose(pv1, pv2)
    assert torch.allclose(ph1, ph2)
    assert not torch.allclose(cond1, cond2)


def test_future_loss_does_not_train_condition_queries():
    pred = _tiny_predictor()
    pv, ph, cond = pred(torch.randn(1, 3, 4, 8), torch.randn(1, 3, 3, 8))
    (pv.sum() + ph.sum()).backward()
    query_grad = pred.condition_query.grad
    proj_grad = pred.out_condition.weight.grad
    assert query_grad is None or torch.count_nonzero(query_grad) == 0
    assert proj_grad is None or torch.count_nonzero(proj_grad) == 0
    assert cond is not None


def test_condition_loss_trains_condition_queries():
    pred = _tiny_predictor()
    _, _, cond = pred(torch.randn(1, 3, 4, 8), torch.randn(1, 3, 3, 8))
    cond.sum().backward()
    assert pred.condition_query.grad is not None
    assert pred.condition_query.grad.abs().sum() > 0
    assert pred.out_condition.weight.grad is not None


def test_condition_tokens_depend_on_context():
    pred = _tiny_predictor()
    pred.eval()
    visual = torch.randn(1, 3, 4, 8)
    tactile = torch.randn(1, 3, 3, 8)
    _, _, c1 = pred(visual, tactile)
    _, _, c2 = pred(visual + 1.0, tactile)
    assert not torch.allclose(c1, c2)
