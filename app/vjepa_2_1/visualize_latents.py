"""视触世界模型的有限隐空间检查。

模型没有像素解码器，所以这里画的是 *token*：V-JEPA / 触觉格子的 PCA
（与 V-JEPA 2.1 teaser 同一思路）、空间余弦误差热图，以及复制最后一帧基线。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from .data import RobotVisionTactileDataset
from .train_tactile_alignment import _run_visual, _visual_tokens, build_models, future_offsets_from_config


def factor_hw(count: int) -> tuple[int, int]:
    """展平空间格子的最近因子对，高在前。"""
    if count <= 0:
        raise ValueError("token count must be positive")
    height = int(count ** 0.5)
    while height > 1 and count % height:
        height -= 1
    return height, count // height


def pca_basis(tokens: torch.Tensor, k: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """在展平 token ``[..., D]`` 上拟合 k 个主成分，返回 ``(mean, pcs)``。"""
    flat = F.layer_norm(tokens, (tokens.shape[-1],)).reshape(-1, tokens.shape[-1]).float()
    mean = flat.mean(0, keepdim=True)
    centered = flat - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return mean, vh[:k]


def pca_maps(tokens: torch.Tensor, mean: torch.Tensor, pcs: torch.Tensor,
             hw: tuple[int, int] | None = None) -> torch.Tensor:
    """把 token 投成 ``[T, H, W, 3]``、取值 ``[0, 1]`` 的 PCA 伪彩图。"""
    if tokens.ndim != 3:
        raise ValueError(f"expected [T, N, D] tokens, got {tuple(tokens.shape)}")
    time, count, dim = tokens.shape
    height, width = hw or factor_hw(count)
    if height * width != count:
        raise ValueError(f"grid {height}x{width} does not match {count} tokens")
    z = F.layer_norm(tokens, (dim,)).reshape(-1, dim).float()
    rgb = (z - mean) @ pcs.T
    lo = torch.quantile(rgb, 0.05, dim=0)
    hi = torch.quantile(rgb, 0.95, dim=0)
    rgb = ((rgb - lo) / (hi - lo + 1e-6)).clamp(0, 1)
    return rgb.view(time, height, width, pcs.shape[0])


def cosine_maps(pred: torch.Tensor, target: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    """逐 token 余弦误差 ``1 - cos``，形状 ``[T, H, W]``，取值 ``[0, 1]``。"""
    cos = F.cosine_similarity(pred, target, dim=-1).clamp(-1, 1)
    return (1 - cos).view(pred.shape[0], hw[0], hw[1])


def step_cosine(pred: torch.Tensor, target: torch.Tensor) -> list[float]:
    """每个时间步的平均余弦相似度。"""
    cos = F.cosine_similarity(pred, target, dim=-1).mean(dim=-1)
    return [float(x) for x in cos]


def copy_last(context: torch.Tensor, horizon: int) -> torch.Tensor:
    return context[-1:].expand(horizon, -1, -1).contiguous()


def _as_image(frames: torch.Tensor) -> torch.Tensor | None:
    """``[T,C,H,W]`` 或 ``[T,H,W,C]`` -> ``[T,3,H,W]``、取值 ``[0,1]``；否则 None。"""
    if frames.ndim != 4:
        return None
    if frames.shape[-1] in (1, 2, 3) and frames.shape[1] > 4:
        frames = frames.permute(0, 3, 1, 2)
    if frames.shape[1] not in (1, 2, 3):
        return None
    if frames.shape[1] == 1:
        frames = frames.repeat(1, 3, 1, 1)
    elif frames.shape[1] == 2:
        frames = torch.cat([frames, frames[:, :1]], dim=1)
    if float(frames.min()) < -0.05 or float(frames.max()) > 1.5:
        mean = frames.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = frames.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        frames = frames * std + mean
    return frames.clamp(0, 1)


def _heat(error: torch.Tensor) -> torch.Tensor:
    """把 ``[H,W]`` 误差场映射成蓝到红的 ``[3,H,W]``。"""
    x = error.clamp(0, 1)
    return torch.stack([x, 0.15 * (1 - x), 1 - x], dim=0)


def _resize(chw: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(chw.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)[0]


def _blank(size: int) -> torch.Tensor:
    return torch.zeros(3, size, size)


def _save_png(path: Path, hw3: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (hw3.clamp(0, 1) * 255).byte().cpu().numpy()
    try:
        from PIL import Image
        Image.fromarray(array).save(path)
        return
    except ImportError:
        pass
    try:
        import cv2
        cv2.imwrite(str(path), array[:, :, ::-1])
        return
    except ImportError as exc:
        raise ImportError("saving PNG requires Pillow or opencv-python") from exc


def render_panel(vision, tactile, vis_gt, vis_pred, tac_gt, tac_pred,
                 vis_err, tac_err, context: int, cell: int = 128,
                 future_indices=None) -> torch.Tensor:
    """把一个 clip 拼成一张 RGB 面板。行是时间，列是视图。"""
    time = vis_gt.shape[0]
    if future_indices is None:
        future_indices = list(range(context, context + (0 if vis_pred is None else vis_pred.shape[0])))
    future_lookup = {int(frame): i for i, frame in enumerate(future_indices)}
    vis_rgb = _as_image(vision)
    tac_rgb = _as_image(tactile)
    rows = []
    for t in range(time):
        cells = []
        cells.append(_resize(vis_rgb[t], cell) if vis_rgb is not None else _blank(cell))
        cells.append(_resize(vis_gt[t].permute(2, 0, 1), cell))
        if t in future_lookup and vis_pred is not None:
            j = future_lookup[t]
            cells.append(_resize(vis_pred[j].permute(2, 0, 1), cell))
            cells.append(_resize(_heat(vis_err[j]), cell))
        else:
            cells.extend([_blank(cell), _blank(cell)])
        cells.append(_resize(tac_rgb[t], cell) if tac_rgb is not None else _blank(cell))
        cells.append(_resize(tac_gt[t].permute(2, 0, 1), cell))
        if t in future_lookup and tac_pred is not None:
            j = future_lookup[t]
            cells.append(_resize(tac_pred[j].permute(2, 0, 1), cell))
            cells.append(_resize(_heat(tac_err[j]), cell))
        else:
            cells.extend([_blank(cell), _blank(cell)])
        rows.append(torch.cat(cells, dim=2))
    return torch.cat(rows, dim=1).permute(1, 2, 0)


@torch.no_grad()
def evaluate_clip(visual, tactile_encoder, predictor, batch, context, horizon,
                  tubelet, device, future_offsets=None) -> dict:
    vision = batch["vision"].to(device)
    tactile = batch["tactile"].to(device)
    if vision.ndim == 5:
        video = vision.permute(0, 2, 1, 3, 4).contiguous()
    else:
        video = vision
    frames = context + horizon
    time_steps = frames // tubelet
    context_steps = context // tubelet
    offsets = tuple(future_offsets or range(tubelet, horizon + 1, tubelet))
    target_idx = [context_steps + x // tubelet - 1 for x in offsets]
    zv = _visual_tokens(_run_visual(visual, video), video.shape[0], time_steps)[0]
    zh = tactile_encoder(tactile)[0]
    vis_hw, tac_hw = factor_hw(zv.shape[1]), factor_hw(zh.shape[1])
    vis_mean, vis_pcs = pca_basis(zv)
    tac_mean, tac_pcs = pca_basis(zh)
    vis_gt = pca_maps(zv, vis_mean, vis_pcs, vis_hw)
    tac_gt = pca_maps(zh, tac_mean, tac_pcs, tac_hw)

    pred_v = pred_h = vis_err = tac_err = None
    metrics = {
        "visual_cosine_pred": None,
        "visual_cosine_copy_last": step_cosine(copy_last(zv[:context_steps], len(offsets)),
                                               zv[target_idx]),
        "tactile_cosine_pred": None,
        "tactile_cosine_copy_last": step_cosine(copy_last(zh[:context_steps], len(offsets)),
                                                zh[target_idx]),
    }
    pv, ph, _ = predictor(zv[:context_steps].unsqueeze(0), zh[:context_steps].unsqueeze(0),
                          future_offsets=[x // tubelet for x in offsets])
    pv, ph = pv[0], ph[0]
    vis_pred = pca_maps(pv, vis_mean, vis_pcs, vis_hw)
    tac_pred = pca_maps(ph, tac_mean, tac_pcs, tac_hw)
    vis_err = cosine_maps(pv, zv[target_idx], vis_hw)
    tac_err = cosine_maps(ph, zh[target_idx], tac_hw)
    metrics["visual_cosine_pred"] = step_cosine(pv, zv[target_idx])
    metrics["tactile_cosine_pred"] = step_cosine(ph, zh[target_idx])
    pred_v, pred_h = vis_pred, tac_pred

    vision_frames = vision[0].cpu() if vision.ndim == 5 else vision.cpu()
    tactile_frames = tactile[0].cpu() if tactile.ndim >= 3 else tactile.cpu()
    # tubelet 编码器每个 tubelet 一行 latent。画每个 tubelet 的终点图像，
    # 让视触面板和 latent 行共用同一时间轴。
    if vision_frames.shape[0] != vis_gt.shape[0]:
        frame_idx = torch.arange(tubelet - 1, vision_frames.shape[0], tubelet)
        vision_frames = vision_frames[frame_idx[:vis_gt.shape[0]]]
    if tactile_frames.ndim >= 1 and tactile_frames.shape[0] != tac_gt.shape[0]:
        frame_idx = torch.arange(tubelet - 1, tactile_frames.shape[0], tubelet)
        tactile_frames = tactile_frames[frame_idx[:tac_gt.shape[0]]]
    panel = render_panel(
        vision_frames, tactile_frames, vis_gt.cpu(),
        None if pred_v is None else pred_v.cpu(),
        tac_gt.cpu(), None if pred_h is None else pred_h.cpu(),
        None if vis_err is None else vis_err.cpu(),
        None if tac_err is None else tac_err.cpu(),
        context=context_steps,
        future_indices=target_idx,
    )
    metrics["visual_beats_copy_last"] = (
        sum(metrics["visual_cosine_pred"]) > sum(metrics["visual_cosine_copy_last"])
    )
    metrics["tactile_beats_copy_last"] = (
        sum(metrics["tactile_cosine_pred"]) > sum(metrics["tactile_cosine_copy_last"])
    )
    metrics["visual_grid"] = list(vis_hw)
    metrics["tactile_grid"] = list(tac_hw)
    return {"metrics": metrics, "panel": panel}


def main():
    p = argparse.ArgumentParser(description="对 latent 预测做 PCA / 误差可视化")
    p.add_argument("--config", required=True)
    p.add_argument("--vjepa-checkpoint", required=True)
    p.add_argument("--resume", required=True, help="训练得到的触觉 + 预测器 checkpoint")
    p.add_argument("--output", default="outputs/tactile_vis")
    p.add_argument("--num-clips", type=int, default=4)
    p.add_argument("--cell", type=int, default=128)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    visual, tactile, predictor, context, horizon, tubelet, _, _ = build_models(
        cfg, args.vjepa_checkpoint, device)
    state = torch.load(args.resume, map_location=device)
    if "tactile" in state:
        tactile.load_state_dict(state["tactile"])
    if "predictor" in state:
        predictor.load_state_dict(state["predictor"])
    visual.eval()
    tactile.eval()
    predictor.eval()
    data_cfg = cfg.get("data", {})
    ds = RobotVisionTactileDataset(
        cfg["manifest"], context, horizon, random_crop=False,
        vision_size=cfg.get("crop_size", data_cfg.get("crop_size", 256)),
        normalize_vision=cfg.get("normalize_vision", data_cfg.get("normalize_vision", True)),
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for i in range(min(args.num_clips, len(ds))):
        batch = ds[i]
        batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in batch.items()}
        result = evaluate_clip(visual, tactile, predictor, batch, context, horizon, tubelet, device,
                              future_offsets=future_offsets_from_config(cfg, tubelet, horizon))
        _save_png(out / f"clip_{i:03d}.png", result["panel"])
        (out / f"clip_{i:03d}.json").write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")
        summary.append(result["metrics"])
        print(f"clip {i}: vis pred {result['metrics']['visual_cosine_pred']} "
              f"copy-last {result['metrics']['visual_cosine_copy_last']} "
              f"beats={result['metrics']['visual_beats_copy_last']}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
