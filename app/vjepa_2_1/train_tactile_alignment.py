"""在冻结的 V-JEPA 2.1 上训练触觉编码器和未来 latent 预测器。

视觉编码器是预训练 V-JEPA 2.1 checkpoint，其最后一层时空 token 就是预测目标隐空间。
另有 tubelet ViT 把触觉编到对齐的时间网格。预测器吃当前视触 token，为每个
``future_offsets`` 吐出一个 latent waypoint，空间维不池化。数据集 ``horizon``
是数据窗口的最大未来长度，可以长于稀疏 query 个数。
"""
import copy
import random
from pathlib import Path
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .data import RobotVisionTactileDataset
from .models.tactile_encoder import TactileEncoder
from .models.tactile_alignment import TactileAlignment
from .models.multimodal_predictor import FutureLatentPredictor


def window_from_config(cfg):
    """从训练配置读取观测长度 n 和预测长度 m。

    规范键是 ``context`` / ``horizon``。``input_frames`` / ``output_frames`` 是别名。
    两者都必须是 ``tubelet_size`` 的正整数倍。
    """
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    configured_visual_tubelet = cfg.get("visual_tubelet_size")
    tubelet = int(model_cfg.get("tubelet_size", configured_visual_tubelet or 1))
    if configured_visual_tubelet is not None and int(configured_visual_tubelet) != tubelet:
        raise ValueError(
            "model.tubelet_size and visual_tubelet_size disagree; configure one visual time grid"
        )
    context = cfg.get("input_frames", data_cfg.get("input_frames",
                      cfg.get("context", data_cfg.get("context"))))
    horizon = cfg.get("output_frames", data_cfg.get("output_frames",
                      cfg.get("horizon", data_cfg.get("horizon"))))
    if context is None or horizon is None:
        raise ValueError("set context/horizon (or input_frames/output_frames) in the config")
    context, horizon = int(context), int(horizon)
    if tubelet <= 0:
        raise ValueError("tubelet_size must be positive")
    if context <= 0 or horizon <= 0:
        raise ValueError("context (n) and horizon (m) must be positive")
    if context % tubelet or horizon % tubelet:
        raise ValueError(
            f"context={context} and horizon={horizon} must be multiples of tubelet_size={tubelet}"
        )
    return context, horizon, tubelet


def future_offsets_from_config(cfg, tubelet, horizon):
    """返回相对 context 边界的 raw-frame 未来 waypoint。"""
    offsets = cfg.get("future_offsets", cfg.get("prediction_offsets"))
    if offsets is None:
        # 视觉编码器每个 tubelet 只出一个 latent。默认 offset 必须落在这条
        # latent 时间格上；tubelet_size=2 时 1、3、5 这种 raw-frame 偏移非法。
        offsets = list(range(tubelet, horizon + 1, tubelet))
    offsets = tuple(int(x) for x in offsets)
    if not offsets or tuple(sorted(set(offsets))) != offsets or offsets[0] <= 0:
        raise ValueError("future_offsets must be a sorted list of unique positive integers")
    if offsets[-1] > horizon:
        raise ValueError("future_offsets cannot exceed horizon")
    if any(x % tubelet for x in offsets):
        raise ValueError("future_offsets must be multiples of tubelet_size")
    return offsets


def sample_future_offsets(max_horizon, count, tubelet=1, min_offset=None, rng=None):
    """在 latent 时间格上采样互不相同、已排序的未来 query 时刻。"""
    max_horizon, count, tubelet = int(max_horizon), int(count), int(tubelet)
    min_offset = tubelet if min_offset is None else int(min_offset)
    if min_offset <= 0 or min_offset % tubelet:
        raise ValueError("random_future_min_offset must be a positive multiple of tubelet_size")
    candidates = list(range(min_offset, max_horizon + 1, tubelet))
    if count <= 0 or count > len(candidates):
        raise ValueError("random future query count is outside the available horizon")
    sampler = random if rng is None else rng
    return tuple(sorted(sampler.sample(candidates, count)))


def _visual_tokens(z, batch, timesteps):
    """把 V-JEPA 输出（[B,N,D] 或嵌套 list）整理成 [B,T,Nt,D]。"""
    while isinstance(z, (list, tuple)):
        if not z:
            raise ValueError("visual encoder returned an empty output")
        z = z[-1]
    if z.ndim == 3:
        if z.shape[0] != batch:
            raise ValueError("visual latent batch dimension does not match input")
        if z.shape[1] % timesteps:
            # 部分 ViT 变体会在前面加一个 CLS token。
            if (z.shape[1] - 1) % timesteps == 0:
                z = z[:, 1:]
            else:
                raise ValueError("visual latent cannot be reshaped into timestep tokens")
        z = z.reshape(batch, timesteps, z.shape[1] // timesteps, z.shape[2])
    elif z.ndim != 4:
        raise ValueError(f"visual latent must be [B,N,D] or [B,T,N,D], got {tuple(z.shape)}")
    if z.shape[0] != batch or z.shape[1] != timesteps:
        raise ValueError(
            f"visual latent time shape {tuple(z.shape[:2])} does not match (batch,timesteps)=({batch},{timesteps})"
        )
    return z


def _run_visual(encoder, video):
    """从 VisionTransformer 或 MultiSeqWrapper 取最后一层 token。

    必须 ``training_mode=False``：V-JEPA 2.1 在 ``training=True`` 时会拼接多层特征，
    那是预训练头，不是这里要预测的时空 latent。
    """
    if hasattr(encoder, "backbone"):
        return encoder([video], masks=None, training_mode=False)
    return encoder(video, training=False)


def _encode_tactile_framewise(encoder, tactile):
    """每帧触觉用单独的时间上下文编码。

    ``TactileEncoder`` 里有时间自注意力。部署时预测器只看当前一帧触觉，因此未来
    target latent 也必须用同样的单帧协议，不能让 target 编码器看到其他（含未来）帧。
    """
    if tactile.ndim == 5:  # 图像 [B,T,C,H,W]
        batch, time, channels, height, width = tactile.shape
        flat = tactile.reshape(batch * time, channels, height, width)
        encoded = encoder(flat)  # [B*T,1,N,D]
        if encoded.ndim != 4 or encoded.shape[1] != 1:
            raise ValueError("framewise tactile encoder must return [B*T,1,N,D]")
        return encoded.reshape(batch, time, encoded.shape[2], encoded.shape[3])
    if tactile.ndim == 3:  # 向量/taxel [B,T,F]
        batch, time, features = tactile.shape
        flat = tactile.reshape(batch * time, 1, features)
        encoded = encoder(flat)
        if encoded.ndim != 4 or encoded.shape[1] != 1:
            raise ValueError("framewise tactile encoder must return [B*T,1,N,D]")
        return encoded.reshape(batch, time, encoded.shape[2], encoded.shape[3])
    raise ValueError(f"tactile batch must be [B,T,C,H,W] or [B,T,F], got {tuple(tactile.shape)}")


def _token_mse(pred, target):
    return F.mse_loss(F.normalize(pred, dim=-1), F.normalize(target, dim=-1))


@torch.no_grad()
def update_ema(online, target, momentum):
    for p, q in zip(online.parameters(), target.parameters()):
        q.data.mul_(momentum).add_(p.data, alpha=1.0 - momentum)
    for b, c in zip(online.buffers(), target.buffers()):
        c.copy_(b)


class MultimodalTrainer:
    def __init__(self, visual_encoder, tactile_encoder, predictor, device="cuda",
                 lr_tactile=1e-4, lr_predictor=3e-4, ema=0.996,
                 visual_dim=None, alignment_projection_dim=256,
                 freeze_visual=True, visual_tubelet_size=1,
                 lambda_global=1.0, lambda_latent=0.3,
                 lambda_temporal=0.1, lambda_future_visual=1.0,
                 lambda_future_tactile=1.0, temperature=0.07,
                 future_offsets=None, state_encoder=None,
                 random_future_offsets=False, random_future_count=None,
                 random_future_min_offset=None, random_seed=0):
        if not 0.0 < ema < 1.0:
            raise ValueError("ema must be between 0 and 1")
        if visual_tubelet_size <= 0:
            raise ValueError("visual_tubelet_size must be positive")
        self.device, self.ema = device, ema
        self.start_epoch = 0
        self.freeze_visual = freeze_visual
        self.visual_tubelet_size = visual_tubelet_size
        self.loss_weights = (lambda_global, lambda_latent, lambda_temporal)
        self.lambda_future_visual = lambda_future_visual
        self.lambda_future_tactile = lambda_future_tactile
        configured_offsets = future_offsets if future_offsets is not None else range(
            visual_tubelet_size,
            visual_tubelet_size * (1 + getattr(predictor, "horizon", 1)),
            visual_tubelet_size,
        )
        self.future_offsets = tuple(configured_offsets)
        if not self.future_offsets:
            raise ValueError("future_offsets must contain at least one waypoint")
        if tuple(sorted(set(self.future_offsets))) != self.future_offsets:
            raise ValueError("future_offsets must be sorted and unique")
        if any(x <= 0 or x % visual_tubelet_size for x in self.future_offsets):
            raise ValueError("future_offsets must be positive multiples of visual_tubelet_size")
        self.random_future_offsets = bool(random_future_offsets)
        self.random_future_count = int(random_future_count or len(self.future_offsets))
        self.random_future_min_offset = random_future_min_offset
        # 私有 RNG：每个 batch 采样不同，固定 seed 时整段训练序列可复现。
        self.future_rng = random.Random(int(random_seed))
        self.state_encoder = state_encoder.to(device) if state_encoder is not None else None
        self.visual = visual_encoder.to(device)
        self.tactile, self.predictor = tactile_encoder.to(device), predictor.to(device)
        if getattr(self.tactile, "tubelet_size", 1) != 1:
            raise ValueError(
                "single-frame tactile context requires tactile.tubelet_size=1"
            )
        self.visual_target = copy.deepcopy(self.visual).to(device).eval()
        for p in self.visual_target.parameters():
            p.requires_grad_(False)
        if freeze_visual:
            self.visual.eval()
            for p in self.visual.parameters():
                p.requires_grad_(False)
        else:
            self.visual.train()
        self.tactile_target = copy.deepcopy(self.tactile).to(device).eval()
        for p in self.tactile_target.parameters():
            p.requires_grad_(False)
        visual_dim = visual_dim or getattr(
            visual_encoder, "embed_dim",
            getattr(getattr(visual_encoder, "backbone", None), "embed_dim", None),
        )
        if visual_dim is None:
            raise ValueError("visual_dim must be provided when the visual encoder exposes no embed_dim")
        self.align = TactileAlignment(tactile_encoder.embed_dim, visual_dim,
                                      projection_dim=alignment_projection_dim,
                                      temperature=temperature).to(device)
        groups = [
            {"params": self.tactile.parameters(), "lr": lr_tactile},
            {"params": self.predictor.parameters(), "lr": lr_predictor},
            {"params": self.align.parameters(), "lr": lr_tactile},
        ]
        if self.state_encoder is not None:
            groups.append({"params": self.state_encoder.parameters(), "lr": lr_tactile})
        if not freeze_visual:
            groups.append({"params": self.visual.parameters(), "lr": lr_tactile * 0.1})
        self.opt = torch.optim.AdamW(groups, weight_decay=0.05)
        self.scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))

    def step(self, batch, context, horizon, stage="joint"):
        if stage not in ("align", "joint"):
            raise ValueError("stage must be 'align' or 'joint'")
        if context <= 0 or horizon <= 0 or context % self.visual_tubelet_size or horizon % self.visual_tubelet_size:
            raise ValueError("context and horizon must be positive multiples of visual_tubelet_size")
        tactile = batch["tactile"].to(self.device)
        video = batch["vision"].to(self.device)
        if video.ndim == 5:  # [B,T,C,H,W] -> V-JEPA 的 [B,C,T,H,W]
            video = video.permute(0, 2, 1, 3, 4).contiguous()
        frames = context + horizon
        time_steps = frames // self.visual_tubelet_size
        with torch.no_grad():
            zv_target = _visual_tokens(_run_visual(self.visual_target, video), video.shape[0], time_steps)
        if self.freeze_visual:
            # 历史单独编码。若先编整段再切片，V-JEPA 时间注意力会泄漏未来信息。
            with torch.no_grad():
                zv_context = _visual_tokens(
                    _run_visual(self.visual_target, video[:, :, :context]),
                    video.shape[0], context // self.visual_tubelet_size)
        else:
            zv_context = _visual_tokens(
                _run_visual(self.visual, video[:, :, :context]),
                video.shape[0], context // self.visual_tubelet_size)
        # 部署触觉通路故意做成帧局部：只用 context 边界上的当前触觉。未来 target
        # 帧也用同一套单帧协议独立编码，让预测器和观测 latent 落在同一分布。
        zh = _encode_tactile_framewise(self.tactile, tactile[:, context - 1:context])
        with torch.no_grad():
            zh_target = _encode_tactile_framewise(self.tactile_target, tactile)
        # 视觉 V-JEPA 可能用时间 tubelet（例如 2 帧），部署触觉编码器则固定 tubelet=1。
        # 触觉 target 留在原生帧网格，下面按每个视觉未来 waypoint 取对应终点帧。
        context_steps = context // self.visual_tubelet_size
        query_offsets = self.future_offsets
        if self.random_future_offsets:
            query_offsets = sample_future_offsets(
                horizon, self.random_future_count, self.visual_tubelet_size,
                self.random_future_min_offset, rng=self.future_rng)
        if len(query_offsets) <= 0:
            raise ValueError("at least one future query is required")
        # 当前触觉帧与同一终点时刻的视觉 latent 对齐。单帧触觉时时间一致性损失
        # 故意为 0，动力学信号由未来 token 预测提供。
        loss_align, metrics = self.align(
            zh, zv_context[:, context_steps - 1:context_steps],
            lambda_global=self.loss_weights[0],
            lambda_latent=self.loss_weights[1],
            lambda_temporal=self.loss_weights[2])
        if stage == "align":
            loss = loss_align
        else:
            state = batch.get("state")
            if self.state_encoder is not None:
                if state is None:
                    raise ValueError("batch is missing state required by state_encoder")
                state_raw = state[:, :context].to(self.device)
                state = self.state_encoder(state_raw)
                # V-JEPA 用时间 tubelet 时，状态取每个 tubelet 终点，对齐视觉 latent 时间格。
                if self.visual_tubelet_size > 1:
                    state = state[:, self.visual_tubelet_size - 1::self.visual_tubelet_size]
            pv, ph, _ = self.predictor(zv_context[:, :context_steps], zh, state,
                                       future_offsets=[x // self.visual_tubelet_size for x in query_offsets])
            target_idx = [context_steps + offset // self.visual_tubelet_size - 1
                          for offset in query_offsets]
            tv = zv_target[:, target_idx]
            tactile_target_idx = [context + offset - 1 for offset in query_offsets]
            if max(tactile_target_idx) >= zh_target.shape[1]:
                raise ValueError("tactile target sequence is shorter than requested future offsets")
            tt = zh_target[:, tactile_target_idx]
            if pv.shape != tv.shape:
                raise ValueError(f"predicted visual latent {tuple(pv.shape)} != target {tuple(tv.shape)}")
            if ph.shape != tt.shape:
                raise ValueError(f"predicted tactile latent {tuple(ph.shape)} != target {tuple(tt.shape)}")
            loss_vis = _token_mse(pv, tv)
            loss_tac = _token_mse(ph, tt)
            loss = loss_align + self.lambda_future_visual * loss_vis + self.lambda_future_tactile * loss_tac
            metrics["loss_future_visual"] = loss_vis.detach()
            metrics["loss_future_tactile"] = loss_tac.detach()
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(
            [p for group in self.opt.param_groups for p in group["params"] if p.grad is not None], 1.0)
        self.scaler.step(self.opt)
        self.scaler.update()
        update_ema(self.tactile, self.tactile_target, self.ema)
        if not self.freeze_visual:
            update_ema(self.visual, self.visual_target, self.ema)
        metrics["loss"] = loss.detach()
        return metrics

    def state_dict(self):
        return {"visual": self.visual.state_dict(), "visual_target": self.visual_target.state_dict(),
                "tactile": self.tactile.state_dict(), "tactile_target": self.tactile_target.state_dict(),
                "predictor": self.predictor.state_dict(), "align": self.align.state_dict(),
                "state_encoder": self.state_encoder.state_dict() if self.state_encoder is not None else None,
                "optimizer": self.opt.state_dict(), "ema": self.ema,
                "future_rng_state": self.future_rng.getstate()}

    def load_state_dict(self, state):
        if state.get("ema") is not None:
            self.ema = float(state["ema"])
        self.start_epoch = int(state.get("epoch", 0))
        for name, module in (("visual", self.visual), ("visual_target", self.visual_target),
                             ("tactile", self.tactile), ("tactile_target", self.tactile_target),
                             ("predictor", self.predictor), ("align", self.align),
                             ("state_encoder", self.state_encoder)):
            if module is None:
                continue
            # 没有本体状态的 checkpoint 该字段是 ``None``。加载到启用 state 的模型时
            # 应保留刚初始化的 state encoder。
            if name in state and state[name] is not None:
                if name == "predictor":
                    incompatible = module.load_state_dict(state[name], strict=False)
                    if incompatible.missing_keys or incompatible.unexpected_keys:
                        print(
                            f"warning: predictor load_state_dict "
                            f"missing={list(incompatible.missing_keys)} "
                            f"unexpected={list(incompatible.unexpected_keys)}"
                        )
                else:
                    module.load_state_dict(state[name])
        if "optimizer" in state:
            try:
                self.opt.load_state_dict(state["optimizer"])
            except (ValueError, RuntimeError) as exc:
                print(f"warning: optimizer state was not restored: {exc}")
        if state.get("future_rng_state") is not None:
            self.future_rng.setstate(state["future_rng_state"])

    def fit(self, loader, epochs, context, horizon, stage="joint", checkpoint_dir=None):
        """跑参考训练循环；调用方可自行包 DDP/AMP。"""
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if stage not in ("align", "joint"):
            raise ValueError("stage must be 'align' or 'joint'")
        for epoch in range(self.start_epoch, self.start_epoch + epochs):
            self.tactile.train()
            self.predictor.train()
            self.align.train()
            sums, count = {}, 0
            for batch in loader:
                metrics = self.step(batch, context, horizon, stage)
                count += 1
                for k, value in metrics.items():
                    sums[k] = sums.get(k, 0.0) + float(value)
            averages = {k: v / max(count, 1) for k, v in sums.items()}
            if checkpoint_dir is not None:
                out = Path(checkpoint_dir)
                out.mkdir(parents=True, exist_ok=True)
                torch.save({"epoch": epoch + 1, **self.state_dict()}, out / f"checkpoint_{epoch + 1:04d}.pt")
            print(f"epoch {epoch + 1}/{epochs}: {averages}")
        return averages


def build_loader(manifest, batch_size=8, context=8, horizon=4, workers=4,
                 require_state=False, vision_size=None, normalize_vision=True):
    ds = RobotVisionTactileDataset(
        manifest, context, horizon, require_state=require_state,
        vision_size=vision_size, normalize_vision=normalize_vision,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers,
                      pin_memory=True, drop_last=False)


def _load_checkpoint(module, path, keys=("encoder", "target_encoder")):
    if not path:
        return
    state = torch.load(path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    for key in keys:
        if key in state:
            state = state[key]
            break
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    if hasattr(module, "backbone") and state and not any(k.startswith("backbone.") for k in state):
        state = {"backbone." + k: v for k, v in state.items()}
    module.load_state_dict(state, strict=False)


def build_models(cfg, vjepa_checkpoint, device="cpu"):
    """按配置构造冻结 V-JEPA、触觉编码器和未来预测器。"""
    from .utils import init_video_model
    m, t = cfg.get("model", {}), cfg.get("tactile", {})
    d = cfg.get("data", {})
    context, horizon, tubelet = window_from_config(cfg)
    future_offsets = future_offsets_from_config(cfg, tubelet, horizon)
    latent_context = context // tubelet
    latent_horizon = horizon // tubelet
    max_context = int(m.get("max_context", cfg.get("max_context", latent_context)))
    if max_context < latent_context:
        raise ValueError(f"max_context={max_context} is shorter than context/tubelet={latent_context}")
    visual, _ = init_video_model(device=device, model_name=m.get("model_name", "vit_large"),
        patch_size=m.get("patch_size", 16), max_num_frames=context + horizon,
        tubelet_size=tubelet, crop_size=cfg.get("crop_size", d.get("crop_size", 256)),
        pred_depth=1, pred_embed_dim=384, use_rope=m.get("use_rope", True),
        modality_embedding=True)
    _load_checkpoint(visual, vjepa_checkpoint)
    dim = getattr(visual, "embed_dim", getattr(getattr(visual, "backbone", None), "embed_dim", 1024))
    taxel_grid = t.get("taxel_grid")
    tactile_tubelet = int(t.get("tubelet_size", 1))
    if tactile_tubelet != 1:
        raise ValueError(
            "this trainer uses a single current GelSight frame for deployment; "
            "tactile.tubelet_size must be 1"
        )
    tactile = TactileEncoder(
        embed_dim=t.get("embed_dim", 384), in_channels=t.get("in_channels", 3),
        # 在线修正每次只看一帧触觉。即使 V-JEPA 用时间 tubelet，这条流也保持帧网格；
        # 训练器按对应 raw-frame 下标取触觉 target。
        patch_size=t.get("patch_size", 16), tubelet_size=1,
        img_size=t.get("img_size"), num_frames=context + horizon,
        depth=t.get("depth", 6), num_heads=t.get("num_heads"),
        input_dim=t.get("input_dim"),
        taxel_grid=tuple(taxel_grid) if taxel_grid is not None else None,
        use_rope=t.get("use_rope", True), backbone=t.get("backbone"))
    state_cfg = cfg.get("state", {})
    state_dim = state_cfg.get("dim")
    state_encoder = None
    if state_dim is not None:
        state_encoder = torch.nn.Sequential(
            torch.nn.Linear(int(state_dim), int(m.get("predictor_dim", 512))),
            torch.nn.GELU(), torch.nn.Linear(int(m.get("predictor_dim", 512)), int(m.get("predictor_dim", 512))))
    predictor = FutureLatentPredictor(
        dim=dim, visual_dim=dim, tactile_dim=tactile.embed_dim,
        hidden_dim=m.get("predictor_dim", 512), depth=m.get("predictor_depth", 6),
        heads=m.get("predictor_heads", 8), horizon=latent_horizon,
        max_context=max_context, future_offsets=[x // tubelet for x in future_offsets],
        max_future_offset=latent_horizon,
        state_dim=int(m.get("predictor_dim", 512)) if state_dim is not None else None,
        num_condition_tokens=m.get("num_condition_tokens", 16),
        condition_out_dim=m.get("condition_out_dim"))
    return visual, tactile, predictor, context, horizon, tubelet, future_offsets, state_encoder


def main():
    p = argparse.ArgumentParser(description="训练触觉对齐 / 未来 latent 模型")
    p.add_argument("--config", required=True)
    p.add_argument("--vjepa-checkpoint", required=True)
    p.add_argument("--resume", default=None)
    p.add_argument("--output", default="outputs/tactile")
    p.add_argument("--stage", choices=("align", "joint"), default="align",
                   help="align 热身触觉编码器；joint 训练未来预测器")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    visual, tactile, predictor, context, horizon, tubelet, future_offsets, state_encoder = build_models(
        cfg, args.vjepa_checkpoint, device)
    loss_cfg = cfg.get("loss", {})
    opt_cfg = cfg.get("optimization", {})
    unfreeze_visual = bool(cfg.get("joint_unfreeze_visual", False)) and args.stage == "joint"
    trainer = MultimodalTrainer(visual, tactile, predictor, device=device,
        lr_tactile=opt_cfg.get("lr_tactile", 1e-4),
        lr_predictor=opt_cfg.get("lr_predictor", 3e-4),
        ema=opt_cfg.get("ema", 0.996), freeze_visual=not unfreeze_visual,
        visual_tubelet_size=tubelet,
        lambda_global=loss_cfg.get("lambda_global", 1.0),
        lambda_latent=loss_cfg.get("lambda_latent", 0.3),
        lambda_temporal=loss_cfg.get("lambda_temporal", 0.1),
        lambda_future_visual=loss_cfg.get("lambda_future_visual", 1.0),
        lambda_future_tactile=loss_cfg.get("lambda_future_tactile", 1.0),
        temperature=loss_cfg.get("temperature", 0.07),
        future_offsets=future_offsets, state_encoder=state_encoder,
        random_future_offsets=cfg.get("random_future_offsets", False),
        random_future_count=cfg.get("random_future_count", len(future_offsets)),
        random_future_min_offset=cfg.get("random_future_min_offset"),
        random_seed=cfg.get("seed", cfg.get("random_seed", 0)))

    if args.resume:
        trainer.load_state_dict(torch.load(args.resume, map_location=device))
    data_cfg = cfg.get("data", {})
    loader = build_loader(
        cfg["manifest"], cfg.get("batch_size", 8), context, horizon,
        cfg.get("num_workers", 4), require_state=state_encoder is not None,
        vision_size=cfg.get("crop_size", data_cfg.get("crop_size", 256)),
        normalize_vision=cfg.get("normalize_vision", data_cfg.get("normalize_vision", True)),
    )
    trainer.fit(loader, cfg.get("epochs", 50), context, horizon, args.stage, args.output)


if __name__ == "__main__":
    main()
