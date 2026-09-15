"""在冻结的 V-JEPA 2.1 上训练触觉编码器和未来 latent 预测器。

视觉编码器是预训练 V-JEPA 2.1 checkpoint，其最后一层时空 token 就是预测目标隐空间。
另有 tubelet ViT 把触觉编到对齐的时间网格。预测器吃当前视触 token，为每个
``future_offsets`` 吐出一个 latent waypoint，空间维不池化。数据集 ``horizon``
是数据窗口的最大未来长度，可以长于稀疏 query 个数。
"""
import copy
import json
import math
import random
import re
import sys
from pathlib import Path
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.vjepa_2_1.data import RobotVisionTactileDataset
from app.vjepa_2_1.models.tactile_encoder import TactileEncoder
from app.vjepa_2_1.models.tactile_alignment import TactileAlignment
from app.vjepa_2_1.models.multimodal_predictor import FutureLatentPredictor


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


def split_indices(n, val_ratio=0.2, seed=0):
    """把 ``n`` 条 clip 随机划成互斥的训练 / 验证下标。"""
    n = int(n)
    val_ratio = float(val_ratio)
    if n <= 0:
        raise ValueError("dataset is empty")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    indices = list(range(n))
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    if val_ratio <= 0.0 or n < 2:
        return sorted(indices), []
    n_val = int(round(n * val_ratio))
    n_val = min(max(1, n_val), n - 1)
    val_idx = sorted(indices[:n_val])
    train_idx = sorted(indices[n_val:])
    return train_idx, val_idx


def output_from_config(cfg, stage, override=None):
    """按当前 stage 选择 checkpoint 输出目录。

    配置写成 ``training.output.align`` / ``training.output.joint``。
    ``--output`` 仍可临时覆盖当前阶段。旧的字符串 ``training.output`` 仅作回退。
    """
    if override:
        return override
    if stage not in ("align", "joint"):
        raise ValueError("stage must be 'align' or 'joint'")
    train_cfg = cfg.get("training", {})
    output = train_cfg.get("output")
    if isinstance(output, dict):
        path = output.get(stage)
        if not path:
            raise ValueError(f"set training.output.{stage} to a checkpoint directory")
        return path
    staged = train_cfg.get(f"output_{stage}")
    if staged:
        return staged
    if isinstance(output, str) and output:
        return output
    raise ValueError("set training.output.align and training.output.joint (or pass --output)")


_OUTPUT_RUN_SUFFIX = re.compile(r"^(?P<stem>.*)\((?P<n>\d+)\)$")


def _output_stem(name):
    match = _OUTPUT_RUN_SUFFIX.match(name)
    if match:
        return match.group("stem"), int(match.group("n"))
    return name, 0


def _existing_output_indices(parent, stem):
    """已有输出目录的编号：基础目录为 0，``name(n)`` 为 n。"""
    if not parent.is_dir():
        return []
    indices = []
    if (parent / stem).exists():
        indices.append(0)
    for child in parent.iterdir():
        match = _OUTPUT_RUN_SUFFIX.match(child.name)
        if match and match.group("stem") == stem:
            indices.append(int(match.group("n")))
    return indices


def unique_output_dir(path, resume=None):
    """若 ``name`` / ``name(n)`` 已存在，则新建 ``name(max+1)``。

    从某一轮目录里的 checkpoint resume 时仍写入该目录。
    """
    path = Path(path)
    parent = path.parent
    stem, _ = _output_stem(path.name)
    if resume:
        resume_dir = Path(resume).resolve().parent
        resume_stem, _ = _output_stem(resume_dir.name)
        if resume_stem == stem and resume_dir.parent.resolve() == parent.resolve():
            return str(resume_dir)
    existing = _existing_output_indices(parent, stem)
    if not existing:
        return str(path)
    return str(parent / f"{stem}({max(existing) + 1})")


def should_save_epoch(completed_epoch, save_every, is_last=False):
    save_every = max(1, int(save_every))
    return bool(is_last) or int(completed_epoch) % save_every == 0


class WarmupCosineMultiplier:
    """线性 warmup，再余弦退火到 ``min_lr_scale * base_lr``。各 param group 保持相对学习率。"""

    def __init__(self, optimizer, total_steps, warmup_steps=0, start_lr_scale=0.01,
                 min_lr_scale=0.01):
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if not 0.0 <= start_lr_scale <= 1.0:
            raise ValueError("start_lr_scale must be in [0, 1]")
        if not 0.0 <= min_lr_scale <= 1.0:
            raise ValueError("min_lr_scale must be in [0, 1]")
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = min(int(warmup_steps), self.total_steps)
        self.start_lr_scale = float(start_lr_scale)
        self.min_lr_scale = float(min_lr_scale)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self._step = 0
        self._apply()

    def scale(self, step):
        step = int(step)
        if self.warmup_steps > 0 and step < self.warmup_steps:
            progress = (step + 1) / float(self.warmup_steps)
            return self.start_lr_scale + progress * (1.0 - self.start_lr_scale)
        cosine_steps = max(1, self.total_steps - self.warmup_steps)
        t = min(max(0, step - self.warmup_steps), cosine_steps)
        progress = t / float(cosine_steps)
        return self.min_lr_scale + (1.0 - self.min_lr_scale) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _apply(self):
        factor = self.scale(self._step)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * factor
        return factor

    def step(self):
        self._step += 1
        return self._apply()

    def jump_to(self, step):
        self._step = max(0, int(step))
        return self._apply()

    def get_last_lr(self):
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self):
        return {
            "step": self._step,
            "base_lrs": list(self.base_lrs),
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "start_lr_scale": self.start_lr_scale,
            "min_lr_scale": self.min_lr_scale,
        }

    def load_state_dict(self, state):
        self._step = int(state.get("step", 0))
        base_lrs = state.get("base_lrs")
        if base_lrs is not None and len(base_lrs) == len(self.base_lrs):
            self.base_lrs = [float(x) for x in base_lrs]
        self._apply()


def load_tensorboard_scalars(log_dir):
    """读取 TensorBoard 目录里的全部 scalar 曲线。"""
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        return {}
    from tensorboard.backend.event_processing.event_accumulator import SCALARS, EventAccumulator

    accumulator = EventAccumulator(str(log_dir), size_guidance={SCALARS: 0})
    accumulator.Reload()
    scalars = {}
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        scalars[tag] = [(int(event.step), float(event.value)) for event in events]
    return scalars


def _safe_curve_name(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "curve"


def save_tensorboard_curves(log_dir, out_dir):
    """把 TensorBoard 里的每条曲线渲染成 PNG，train/val 叠在同一张图上。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = load_tensorboard_scalars(log_dir)
    if not series:
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    families = {}
    for tag, points in series.items():
        if "/" in tag:
            family, split = tag.rsplit("/", 1)
        else:
            family, split = tag, "value"
        families.setdefault(family, {})[split] = points

    saved = []

    def _draw(ax, points, label, color=None):
        if not points:
            return
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o" if len(points) <= 40 else None, linewidth=1.6, label=label, color=color)

    epoch_families = []
    for family, splits in sorted(families.items()):
        epoch_splits = {k: v for k, v in splits.items() if k != "train_step"}
        if epoch_splits:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for name, points in epoch_splits.items():
                _draw(ax, points, name)
            ax.set_title(family)
            ax.set_xlabel("epoch")
            ax.set_ylabel(family)
            ax.grid(True, alpha=0.3)
            if len(epoch_splits) > 1:
                ax.legend()
            fig.tight_layout()
            path = out_dir / f"{_safe_curve_name(family)}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            saved.append(path)
            epoch_families.append(family)
        if "train_step" in splits:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            _draw(ax, splits["train_step"], "train_step")
            ax.set_title(f"{family} (train step)")
            ax.set_xlabel("step")
            ax.set_ylabel(family)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = out_dir / f"{_safe_curve_name(family)}_train_step.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            saved.append(path)

    if epoch_families:
        cols = 2 if len(epoch_families) > 1 else 1
        rows = (len(epoch_families) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 3.6 * rows), squeeze=False)
        for idx, family in enumerate(epoch_families):
            ax = axes[idx // cols][idx % cols]
            for name, points in families[family].items():
                if name == "train_step":
                    continue
                _draw(ax, points, name)
            ax.set_title(family)
            ax.set_xlabel("epoch")
            ax.grid(True, alpha=0.3)
            if len([k for k in families[family] if k != "train_step"]) > 1:
                ax.legend()
        for idx in range(len(epoch_families), rows * cols):
            axes[idx // cols][idx % cols].axis("off")
        fig.tight_layout()
        path = out_dir / "overview.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
    return saved


_EPOCH_CHECKPOINT = re.compile(r"^checkpoint_(\d+)\.pt$")
_STEP_CHECKPOINT = re.compile(r"^checkpoint_step_(\d+)\.pt$")
_METRICS_FILE = "metrics.json"


def _metrics_path(directory):
    return Path(directory) / _METRICS_FILE


def _read_metrics(directory):
    path = _metrics_path(directory)
    if not path.is_file():
        return {"best": None, "best_loss": None, "records": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"best": None, "best_loss": None, "records": {}}
    if not isinstance(data, dict):
        return {"best": None, "best_loss": None, "records": {}}
    data.setdefault("records", {})
    if not isinstance(data["records"], dict):
        data["records"] = {}
    return data


def record_checkpoint_metrics(directory, filename, epoch, metrics):
    """把本次 checkpoint 的 loss 记进 sidecar，供 joint 选最优权重。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data = _read_metrics(directory)
    loss = None if metrics is None else metrics.get("loss")
    if loss is not None:
        loss = float(loss)
    record = {"epoch": int(epoch), "loss": loss}
    if metrics:
        record["metrics"] = {k: float(v) for k, v in metrics.items()}
    data["records"][filename] = record
    if loss is not None and (data.get("best_loss") is None or loss < float(data["best_loss"])):
        data["best"] = filename
        data["best_loss"] = loss
    _metrics_path(directory).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return data.get("best")


def _log_epoch_curves(writer, train_metrics, val_metrics, epoch):
    if writer is None:
        return
    for key, value in train_metrics.items():
        writer.add_scalar(f"{key}/train", float(value), epoch)
    for key, value in (val_metrics or {}).items():
        writer.add_scalar(f"{key}/val", float(value), epoch)
    writer.flush()


def prune_checkpoints(directory, keep_last=1):
    """只保留 best 和最近 ``keep_last`` 个完整 epoch 文件，删掉其余以节省磁盘。"""
    directory = Path(directory)
    keep_last = max(1, int(keep_last))
    metrics = _read_metrics(directory)
    keep = set()
    best_name = metrics.get("best")
    if best_name:
        keep.add(best_name)
    epoch_ckpts, step_ckpts = list_stage_checkpoints(directory)
    for _, path in epoch_ckpts[-keep_last:]:
        keep.add(path.name)
    removed = []
    for _, path in epoch_ckpts + step_ckpts:
        if path.name in keep:
            continue
        path.unlink(missing_ok=True)
        removed.append(path.name)
    for tmp in directory.glob("checkpoint_*.pt.tmp"):
        tmp.unlink(missing_ok=True)
        removed.append(tmp.name)
    return removed


def _save_checkpoint(trainer, checkpoint_dir, filename, epoch, train_metrics,
                     val_metrics=None, keep_last=1):
    out = Path(checkpoint_dir)
    out.mkdir(parents=True, exist_ok=True)
    selection = val_metrics or train_metrics
    payload = {
        "epoch": epoch,
        "loss": selection.get("loss"),
        "metrics": train_metrics,
        "val_metrics": val_metrics or None,
        **trainer.state_dict(),
    }
    target = out / filename
    tmp = out / (filename + ".tmp")
    try:
        torch.save(payload, tmp)
        tmp.replace(target)
    except Exception:
        tmp.unlink(missing_ok=True)
        if target.exists() and target.stat().st_size < 8 * 1024 * 1024:
            target.unlink(missing_ok=True)
        raise
    best_name = record_checkpoint_metrics(out, filename, epoch, selection)
    prune_checkpoints(out, keep_last=keep_last)
    return best_name


def list_stage_checkpoints(directory):
    directory = Path(directory)
    epoch_ckpts, step_ckpts = [], []
    if not directory.is_dir():
        return epoch_ckpts, step_ckpts
    for path in directory.glob("checkpoint_*.pt"):
        match = _EPOCH_CHECKPOINT.match(path.name)
        if match:
            epoch_ckpts.append((int(match.group(1)), path))
            continue
        match = _STEP_CHECKPOINT.match(path.name)
        if match:
            step_ckpts.append((int(match.group(1)), path))
    epoch_ckpts.sort()
    step_ckpts.sort()
    return epoch_ckpts, step_ckpts


def best_checkpoint(directory):
    """返回目录里最优的训练 checkpoint。

    优先 ``metrics.json`` 记录的最低 loss；没有记录时用最新的
    ``checkpoint_XXXX.pt``，再退到 ``checkpoint_step_XXXX.pt``。
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {directory}")
    metrics = _read_metrics(directory)
    best_name = metrics.get("best")
    if best_name:
        best_path = directory / best_name
        if best_path.is_file():
            return best_path
    epoch_ckpts, step_ckpts = list_stage_checkpoints(directory)
    records = metrics.get("records") or {}
    scored = []
    for epoch, path in epoch_ckpts:
        record = records.get(path.name) or {}
        if record.get("loss") is not None:
            scored.append((float(record["loss"]), -epoch, path))
    if scored:
        scored.sort()
        return scored[0][2]
    if epoch_ckpts:
        return epoch_ckpts[-1][1]
    if step_ckpts:
        return step_ckpts[-1][1]
    raise FileNotFoundError(f"no checkpoints in {directory}")


def resume_from_config(cfg, stage, override=None):
    """joint 默认从 ``training.output.align`` 选最优 checkpoint。

    ``training.resume`` / ``--resume`` 可以是具体路径、``auto``，或 ``null``。
    align 阶段 ``null`` 表示从头训练。
    """
    train_cfg = cfg.get("training", {})
    specified = train_cfg.get("resume") if override is None else override
    if specified is False:
        return None
    if isinstance(specified, str):
        text = specified.strip()
        if text.lower() in ("none", "null", "false"):
            return None
        if text.lower() in ("auto", "true"):
            specified = True
        elif text:
            return specified
        else:
            specified = None
    if specified is None and stage != "joint":
        return None
    align_dir = output_from_config(cfg, "align")
    return str(best_checkpoint(align_dir))


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
        self.scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
        self.scheduler = None
        self._loaded_scheduler = False

    def step(self, batch, context, horizon, stage="joint", train=True):
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
        if self.random_future_offsets and train:
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
        metrics["loss"] = loss.detach()
        if not train:
            return metrics
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
        return metrics

    def state_dict(self):
        payload = {
            "tactile": self.tactile.state_dict(),
            "tactile_target": self.tactile_target.state_dict(),
            "predictor": self.predictor.state_dict(),
            "align": self.align.state_dict(),
            "state_encoder": self.state_encoder.state_dict() if self.state_encoder is not None else None,
            "optimizer": self.opt.state_dict(),
            "ema": self.ema,
            "future_rng_state": self.future_rng.getstate(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
            "freeze_visual": self.freeze_visual,
            # 冻结视觉编码器仍从 V-JEPA checkpoint 加载，不必每个 epoch 再存两份 ViT。
            "visual": None if self.freeze_visual else self.visual.state_dict(),
            "visual_target": None if self.freeze_visual else self.visual_target.state_dict(),
        }
        return payload

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
        self._loaded_scheduler = False
        if self.scheduler is not None and state.get("scheduler") is not None:
            try:
                self.scheduler.load_state_dict(state["scheduler"])
                self._loaded_scheduler = True
            except (ValueError, RuntimeError, TypeError, KeyError) as exc:
                print(f"warning: scheduler state was not restored: {exc}")

    def configure_scheduler(self, total_steps, warmup_steps=0, start_lr_scale=0.01,
                            min_lr_scale=0.01):
        self.scheduler = WarmupCosineMultiplier(
            self.opt, total_steps=total_steps, warmup_steps=warmup_steps,
            start_lr_scale=start_lr_scale, min_lr_scale=min_lr_scale,
        )
        return self.scheduler

    def _set_modules_train(self, train):
        self.tactile.train(train)
        self.predictor.train(train)
        self.align.train(train)
        if self.state_encoder is not None:
            self.state_encoder.train(train)
        if not self.freeze_visual:
            self.visual.train(train)

    @torch.no_grad()
    def evaluate(self, loader, context, horizon, stage="joint"):
        """在验证集上算平均损失，不更新参数。"""
        self._set_modules_train(False)
        sums, count = {}, 0
        for batch in loader:
            metrics = self.step(batch, context, horizon, stage, train=False)
            count += 1
            for k, value in metrics.items():
                sums[k] = sums.get(k, 0.0) + float(value)
        if count <= 0:
            return {}
        return {k: v / count for k, v in sums.items()}

    def fit(self, loader, epochs, context, horizon, stage="joint", checkpoint_dir=None,
            max_steps=None, val_loader=None, log_dir=None, keep_checkpoints=1,
            save_every=10):
        """跑参考训练循环；调用方可自行包 DDP/AMP。"""
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if stage not in ("align", "joint"):
            raise ValueError("stage must be 'align' or 'joint'")
        if max_steps is not None and int(max_steps) <= 0:
            raise ValueError("max_steps must be positive")
        writer = None
        if log_dir is not None:
            from torch.utils.tensorboard import SummaryWriter
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(log_path))
            print(f"tensorboard logdir={log_path}", flush=True)
        remaining = None if max_steps is None else int(max_steps)
        save_every = max(1, int(save_every))
        last_epoch = self.start_epoch + epochs - 1
        global_step = 0
        last_averages = {}
        try:
            for epoch in range(self.start_epoch, self.start_epoch + epochs):
                self._set_modules_train(True)
                sums, count = {}, 0
                for batch in loader:
                    metrics = self.step(batch, context, horizon, stage, train=True)
                    if self.scheduler is not None:
                        self.scheduler.step()
                    count += 1
                    global_step += 1
                    for k, value in metrics.items():
                        sums[k] = sums.get(k, 0.0) + float(value)
                    if writer is not None:
                        for k, value in metrics.items():
                            writer.add_scalar(f"{k}/train_step", float(value), global_step)
                        lrs = self.opt.param_groups
                        writer.add_scalar("lr/tactile", float(lrs[0]["lr"]), global_step)
                        if len(lrs) > 1:
                            writer.add_scalar("lr/predictor", float(lrs[1]["lr"]), global_step)
                    print(
                        f"epoch {epoch + 1} step {count}: "
                        + ", ".join(f"{k}={float(v):.4f}" for k, v in metrics.items()),
                        flush=True,
                    )
                    if remaining is not None:
                        remaining -= 1
                        if remaining <= 0:
                            averages = {k: v / max(count, 1) for k, v in sums.items()}
                            val_averages = self.evaluate(val_loader, context, horizon, stage) if val_loader else {}
                            _log_epoch_curves(writer, averages, val_averages, epoch + 1)
                            if checkpoint_dir is not None:
                                _save_checkpoint(
                                    self, checkpoint_dir, f"checkpoint_step_{count:04d}.pt",
                                    epoch, averages, val_averages, keep_last=keep_checkpoints)
                            print(f"stopped after {count} step(s): {averages}", flush=True)
                            if val_averages:
                                print(f"val: {val_averages}", flush=True)
                            return val_averages or averages
                averages = {k: v / max(count, 1) for k, v in sums.items()}
                val_averages = self.evaluate(val_loader, context, horizon, stage) if val_loader else {}
                _log_epoch_curves(writer, averages, val_averages, epoch + 1)
                if checkpoint_dir is not None and should_save_epoch(
                        epoch + 1, save_every, is_last=(epoch == last_epoch)):
                    best_name = _save_checkpoint(
                        self, checkpoint_dir, f"checkpoint_{epoch + 1:04d}.pt",
                        epoch + 1, averages, val_averages, keep_last=keep_checkpoints)
                    selection = val_averages or averages
                    if best_name == f"checkpoint_{epoch + 1:04d}.pt":
                        print(
                            f"new best checkpoint: {Path(checkpoint_dir) / best_name} "
                            f"loss={selection.get('loss')}",
                            flush=True,
                        )
                print(f"epoch {epoch + 1}/{epochs}: {averages}", flush=True)
                if val_averages:
                    print(f"epoch {epoch + 1}/{epochs} val: {val_averages}", flush=True)
                last_averages = val_averages or averages
        finally:
            if writer is not None:
                writer.flush()
                writer.close()
                curves_dir = Path(log_dir).parent / "curves"
                try:
                    saved = save_tensorboard_curves(log_dir, curves_dir)
                except Exception as exc:
                    print(f"warning: failed to render tensorboard curves: {exc}", flush=True)
                else:
                    if saved:
                        print(f"saved {len(saved)} curve plots to {curves_dir}", flush=True)
        return last_averages


def build_loaders(manifest, batch_size=8, context=8, horizon=4, workers=4,
                  require_state=False, vision_size=None, normalize_vision=True,
                  pin_memory=False, val_ratio=0.2, seed=0):
    """训练前按 seed 随机划分训练 / 验证 clip，验证集不做随机时间裁剪。"""
    full = RobotVisionTactileDataset(
        manifest, context, horizon, require_state=require_state,
        vision_size=vision_size, normalize_vision=normalize_vision,
        random_crop=True,
    )
    train_idx, val_idx = split_indices(len(full), val_ratio=val_ratio, seed=seed)
    train_items = [full.items[i] for i in train_idx]
    train_ds = RobotVisionTactileDataset(
        manifest, context, horizon, require_state=require_state,
        vision_size=vision_size, normalize_vision=normalize_vision,
        random_crop=True, items=train_items,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=pin_memory, drop_last=False,
    )
    val_loader = None
    if val_idx:
        val_items = [full.items[i] for i in val_idx]
        val_ds = RobotVisionTactileDataset(
            manifest, context, horizon, require_state=require_state,
            vision_size=vision_size, normalize_vision=normalize_vision,
            random_crop=False, items=val_items,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=workers,
            pin_memory=pin_memory, drop_last=False,
        )
    split = {
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "n_total": len(full),
        "train_indices": train_idx,
        "val_indices": val_idx,
        "train_items": [item.get("vision") for item in train_items],
        "val_items": [full.items[i].get("vision") for i in val_idx],
    }
    print(
        f"data split: train={len(train_idx)} val={len(val_idx)} "
        f"val_ratio={val_ratio} seed={seed}",
        flush=True,
    )
    return train_loader, val_loader, split


def build_loader(manifest, batch_size=8, context=8, horizon=4, workers=4,
                 require_state=False, vision_size=None, normalize_vision=True,
                 pin_memory=False):
    train_loader, _, _ = build_loaders(
        manifest, batch_size, context, horizon, workers, require_state,
        vision_size, normalize_vision, pin_memory, val_ratio=0.0, seed=0,
    )
    return train_loader


def _load_checkpoint(module, path, keys=("ema_encoder", "encoder", "target_encoder")):
    if not path:
        return
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    for key in keys:
        if key in state:
            print(f"loading visual weights from checkpoint key '{key}'")
            state = state[key]
            break
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    if hasattr(module, "backbone") and state and not any(k.startswith("backbone.") for k in state):
        state = {"backbone." + k: v for k, v in state.items()}
    incompatible = module.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "warning: visual checkpoint load_state_dict "
            f"missing={list(incompatible.missing_keys)[:12]} "
            f"unexpected={list(incompatible.unexpected_keys)[:12]}"
        )


def build_models(cfg, vjepa_checkpoint, device="cpu"):
    """按配置构造冻结 V-JEPA、触觉编码器和未来预测器。"""
    from app.vjepa_2_1.utils import init_video_model
    m, t = cfg.get("model", {}), cfg.get("tactile", {})
    d = cfg.get("data", {})
    context, horizon, tubelet = window_from_config(cfg)
    future_offsets = future_offsets_from_config(cfg, tubelet, horizon)
    latent_context = context // tubelet
    latent_horizon = horizon // tubelet
    max_context = int(m.get("max_context", cfg.get("max_context", latent_context)))
    if max_context < latent_context:
        raise ValueError(f"max_context={max_context} is shorter than context/tubelet={latent_context}")
    crop_size = int(cfg.get("crop_size", d.get("crop_size", 256)))
    patch_size = int(m.get("patch_size", 16))
    visual, _ = init_video_model(
        device=device, model_name=m.get("model_name", "vit_large"),
        patch_size=patch_size, max_num_frames=context + horizon,
        tubelet_size=tubelet, crop_size=crop_size,
        use_rope=m.get("use_rope", True), use_sdpa=m.get("use_sdpa", True),
        modality_embedding=True,
        img_temporal_dim_size=m.get("img_temporal_dim_size", 1),
        interpolate_rope=m.get("interpolate_rope", True),
        build_predictor=False)
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
    visual_tokens = (crop_size // patch_size) ** 2
    tac_size = t.get("img_size") or crop_size
    if isinstance(tac_size, (list, tuple)):
        tac_h, tac_w = int(tac_size[0]), int(tac_size[1])
    else:
        tac_h = tac_w = int(tac_size)
    tac_patch = int(t.get("patch_size", 16))
    tactile_tokens = max(1, (tac_h // tac_patch) * (tac_w // tac_patch))
    predictor = FutureLatentPredictor(
        dim=dim, visual_dim=dim, tactile_dim=tactile.embed_dim,
        hidden_dim=m.get("predictor_dim", 512), depth=m.get("predictor_depth", 6),
        heads=m.get("predictor_heads", 8), horizon=latent_horizon,
        max_context=max_context, future_offsets=[x // tubelet for x in future_offsets],
        max_future_offset=latent_horizon,
        max_visual_tokens=visual_tokens, max_tactile_tokens=tactile_tokens,
        state_dim=int(m.get("predictor_dim", 512)) if state_dim is not None else None,
        num_condition_tokens=m.get("num_condition_tokens", 16),
        condition_out_dim=m.get("condition_out_dim"))
    return visual, tactile, predictor, context, horizon, tubelet, future_offsets, state_encoder


def main():
    p = argparse.ArgumentParser(description="训练触觉对齐 / 未来 latent 模型")
    p.add_argument("--config", required=True)
    # These options remain as optional one-off overrides; normal runs should
    # keep them in the YAML so a stage can be reproduced from its config.
    p.add_argument("--vjepa-checkpoint", default=None)
    p.add_argument("--resume", default=None,
                   help="checkpoint 路径；joint 默认 auto，从 training.output.align 选最优")
    p.add_argument("--output", default=None,
                   help="覆盖当前 stage 的输出目录；默认用 training.output.align/joint")
    p.add_argument("--stage", choices=("align", "joint"), default=None,
                   help="align 热身触觉编码器；joint 训练未来预测器")
    p.add_argument("--max-steps", type=int, default=None,
                   help="只跑这么多 step 后保存并退出，用于冒烟测试")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    train_cfg = cfg.get("training", {})
    vjepa_checkpoint = args.vjepa_checkpoint or train_cfg.get("vjepa_checkpoint")
    if not vjepa_checkpoint:
        p.error("set training.vjepa_checkpoint in the config (or pass --vjepa-checkpoint)")
    stage = args.stage or train_cfg.get("stage", "align")
    if stage not in ("align", "joint"):
        p.error("training.stage must be 'align' or 'joint'")
    try:
        output = output_from_config(cfg, stage, override=args.output)
        resume = resume_from_config(cfg, stage, override=args.resume)
        allocated = unique_output_dir(output, resume=resume)
        if allocated != str(Path(output)):
            print(f"output {output} already exists, writing to {allocated}", flush=True)
        output = allocated
    except (ValueError, FileNotFoundError) as exc:
        p.error(str(exc))
    max_steps = args.max_steps if args.max_steps is not None else train_cfg.get("max_steps")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"device={device} stage={stage} output={output} "
        f"resume={resume or 'none'} checkpoint={vjepa_checkpoint}",
        flush=True,
    )
    visual, tactile, predictor, context, horizon, tubelet, future_offsets, state_encoder = build_models(
        cfg, vjepa_checkpoint, device)
    loss_cfg = cfg.get("loss", {})
    opt_cfg = cfg.get("optimization", {})
    unfreeze_visual = bool(cfg.get("joint_unfreeze_visual", False)) and stage == "joint"
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

    data_cfg = cfg.get("data", {})
    seed = int(cfg.get("seed", cfg.get("random_seed", 0)))
    val_ratio = float(cfg.get("val_ratio", data_cfg.get("val_ratio", 0.2)))
    train_loader, val_loader, split = build_loaders(
        cfg["manifest"], cfg.get("batch_size", 8), context, horizon,
        cfg.get("num_workers", 4), require_state=state_encoder is not None,
        vision_size=cfg.get("crop_size", data_cfg.get("crop_size", 256)),
        normalize_vision=cfg.get("normalize_vision", data_cfg.get("normalize_vision", True)),
        pin_memory=device.startswith("cuda"),
        val_ratio=val_ratio, seed=seed,
    )
    epochs = int(cfg.get("epochs", 50))
    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    warmup_epochs = opt_cfg.get("warmup_epochs", 5)
    warmup_steps = opt_cfg.get("warmup_steps")
    if warmup_steps is None:
        warmup_steps = int(warmup_epochs) * steps_per_epoch
    trainer.configure_scheduler(
        total_steps=total_steps, warmup_steps=int(warmup_steps),
        start_lr_scale=float(opt_cfg.get("start_lr_scale", 0.01)),
        min_lr_scale=float(opt_cfg.get("min_lr_scale", 0.01)),
    )
    print(
        f"lr schedule: warmup_steps={int(warmup_steps)} total_steps={total_steps} "
        f"start_scale={opt_cfg.get('start_lr_scale', 0.01)} "
        f"min_scale={opt_cfg.get('min_lr_scale', 0.01)}",
        flush=True,
    )
    if resume:
        trainer.load_state_dict(torch.load(resume, map_location=device, weights_only=False))
        if not trainer._loaded_scheduler:
            trainer.scheduler.jump_to(trainer.start_epoch * steps_per_epoch)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    log_dir = None if cfg.get("tensorboard", True) is False else str(out_dir / "tensorboard")
    trainer.fit(train_loader, epochs, context, horizon, stage, output,
                max_steps=max_steps, val_loader=val_loader, log_dir=log_dir,
                keep_checkpoints=int(train_cfg.get("keep_checkpoints", cfg.get("keep_checkpoints", 1))),
                save_every=int(train_cfg.get("save_every", cfg.get("save_every", 10))))


if __name__ == "__main__":
    main()
