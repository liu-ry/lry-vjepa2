"""在单卡上计时一次部署路径的世界模型前向。

与推理一致：``context`` 帧视觉历史、一帧触觉、稀疏未来 query。只统计 CUDA event
kernel 时间，不含 dataloader。

    python -m app.vjepa_2_1.benchmark_world_model \
        --config configs/train_2_1/tactile_alignment.yaml \
        --dtype bf16 --warmup 20 --iters 50
"""
from __future__ import annotations

import argparse
import statistics

import torch
import yaml

from .train_tactile_alignment import _run_visual, _visual_tokens, build_models


def _dtype(name: str) -> torch.dtype:
    mapping = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if name not in mapping:
        raise ValueError(f"dtype must be one of {tuple(mapping)}, got {name}")
    return mapping[name]


def _nparams(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@torch.inference_mode()
def _forward(visual, tactile_enc, predictor, state_enc, video, tactile, state, tubelet, offsets):
    batch = video.shape[0]
    context_steps = video.shape[2] // tubelet
    zv = _visual_tokens(_run_visual(visual, video), batch, context_steps)
    zh = tactile_enc(tactile)
    st = state_enc(state) if state_enc is not None and state is not None else None
    if st is not None and tubelet > 1:
        st = st[:, tubelet - 1 :: tubelet]
    return predictor(zv, zh, st, future_offsets=[x // tubelet for x in offsets])


def _time_ms(fn, warmup: int, iters: int, device: str) -> list[float]:
    use_cuda = device.startswith("cuda")
    if use_cuda:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    if use_cuda:
        torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        if use_cuda:
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
        else:
            t0 = torch.perf_counter()
            fn()
            samples.append((torch.perf_counter() - t0) * 1000.0)
    return samples


def _report(name: str, samples: list[float]) -> None:
    mean = statistics.fmean(samples)
    p50 = statistics.median(samples)
    p95 = sorted(samples)[max(0, int(round(0.95 * (len(samples) - 1))))]
    print(f"{name:22s}  mean {mean:7.2f} ms   p50 {p50:7.2f}   p95 {p95:7.2f}   ({1000.0 / mean:5.1f} Hz)")


def main() -> None:
    parser = argparse.ArgumentParser(description="计时一次世界模型推理")
    parser.add_argument("--config", default="configs/train_2_1/tactile_alignment.yaml")
    parser.add_argument("--vjepa-checkpoint", default="", help="可选；测延迟用随机权重即可")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", default="bf16", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _dtype(args.dtype)
    if device == "cpu" and dtype != torch.float32:
        print("warning: CPU 计时回退到 fp32")
        dtype = torch.float32
    visual, tactile, predictor, context, horizon, tubelet, future_offsets, state_enc = build_models(
        cfg, args.vjepa_checkpoint, device
    )
    for module in (visual, tactile, predictor, state_enc):
        if module is None:
            continue
        module.eval()
        module.requires_grad_(False)
    crop = int(cfg.get("crop_size", cfg.get("data", {}).get("crop_size", 256)))
    tac_cfg = cfg.get("tactile", {})
    tac_size = int(tac_cfg.get("img_size") or crop)
    tac_ch = int(tac_cfg.get("in_channels", 3))
    batch = args.batch_size
    video = torch.randn(batch, 3, context, crop, crop, device=device)
    tac = torch.randn(batch, 1, tac_ch, tac_size, tac_size, device=device)
    state = None
    if state_enc is not None:
        state = torch.randn(batch, context, int(cfg["state"]["dim"]), device=device)

    if args.compile and hasattr(torch, "compile"):
        visual, tactile, predictor = (torch.compile(m) for m in (visual, tactile, predictor))
        if state_enc is not None:
            state_enc = torch.compile(state_enc)

    n_vis = _nparams(visual)
    n_tac = _nparams(tactile)
    n_pred = _nparams(predictor)
    print(f"device {device}  {torch.cuda.get_device_name(0) if device.startswith('cuda') else 'cpu'}")
    print(f"dtype {dtype}  batch {batch}  context {context}  offsets {future_offsets}")
    print(
        f"params  visual {n_vis/1e6:.1f}M  tactile {n_tac/1e6:.1f}M  "
        f"predictor {n_pred/1e6:.1f}M  total {(n_vis+n_tac+n_pred)/1e6:.1f}M"
    )

    autocast = torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=dtype, enabled=dtype != torch.float32)

    def visual_step():
        with autocast:
            _visual_tokens(_run_visual(visual, video), batch, context // tubelet)

    def tactile_step():
        with autocast:
            tactile(tac)

    def full_step():
        with autocast:
            _forward(visual, tactile, predictor, state_enc, video, tac, state, tubelet, future_offsets)

    with torch.inference_mode(), autocast:
        zv = _visual_tokens(_run_visual(visual, video), batch, context // tubelet)
        zh = tactile(tac)
        st = state_enc(state) if state_enc is not None else None
        if st is not None and tubelet > 1:
            st = st[:, tubelet - 1 :: tubelet]
        pv, ph, cond = predictor(zv, zh, st, future_offsets=[x // tubelet for x in future_offsets])
    print(f"z_v {tuple(zv.shape)}  z_h {tuple(zh.shape)}")
    print(f"pred_v {tuple(pv.shape)}  pred_h {tuple(ph.shape)}  cond {None if cond is None else tuple(cond.shape)}")
    seq = zv.shape[1] * zv.shape[2] + zh.shape[1] * zh.shape[2]
    seq += pv.shape[1] * (pv.shape[2] + ph.shape[2])
    if cond is not None:
        seq += cond.shape[1]
    print(f"predictor sequence length ~{seq} tokens (context + future queries + condition)")

    lat_off = [x // tubelet for x in future_offsets]

    def pred_only():
        with autocast:
            predictor(zv, zh, st, future_offsets=lat_off)

    with torch.inference_mode():
        _report("visual encoder", _time_ms(visual_step, args.warmup, args.iters, device))
        _report("tactile encoder", _time_ms(tactile_step, args.warmup, args.iters, device))

        _report("predictor", _time_ms(pred_only, args.warmup, args.iters, device))
        _report("full forward", _time_ms(full_step, args.warmup, args.iters, device))


if __name__ == "__main__":
    main()
