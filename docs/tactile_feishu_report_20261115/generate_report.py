#!/usr/bin/env python3
"""Generate data-driven figures and a Feishu-friendly HTML report."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mistune import create_markdown


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FINAL = ROOT / "data/runs/tactile_formal_final_20261115/final_model"
PREV = ROOT / "data/runs/tactile_final_autoresearch_20261115/final_model"
IMAGES = HERE / "images"

BLUE = "#2F6FA5"
TEAL = "#2A9D8F"
ORANGE = "#E76F51"
GOLD = "#E9A23B"
GRAY = "#6B7280"
LIGHT = "#EEF4F8"


def configure_style() -> None:
    font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update(
        {
            "font.family": font_name,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "font.size": 10.5,
            "legend.frameon": False,
            "grid.color": "#E2E8F0",
            "grid.linewidth": 0.8,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(IMAGES / name, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def chart_causal_pipeline() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), gridspec_kw={"height_ratios": [1, 1.25]})
    for ax in axes:
        ax.set_xlim(-0.5, 34.5)
        ax.set_ylim(-1.2, 2.3)
        ax.axis("off")

    def box(ax, x, y, w, h, text, color, edge=None, alpha=1.0, hatch=None):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor=color,
            edgecolor=edge or color,
            linewidth=1.2,
            alpha=alpha,
            hatch=hatch,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="#1F2937")

    ax = axes[0]
    ax.set_title("A. 运行时：只读取 4 帧观测历史", loc="left")
    for i in range(4):
        box(ax, i, 0.55, 0.85, 0.65, f"t-{3-i}" if i < 3 else "t", BLUE, alpha=0.18)
    ax.text(2.0, 1.72, "tactile + vision + state 的观测前缀", ha="center", color=BLUE, weight="bold")
    box(
        ax,
        5.1,
        0.45,
        17.5,
        0.88,
        "✕  未来 30 帧：运行时不可见，不进入 encoder / predictor 条件",
        "#F8FAFC",
        GRAY,
        0.9,
    )
    box(ax, 23.2, 0.55, 9.7, 0.65, "输出：latent + query tokens", TEAL, alpha=0.18)
    ax.add_patch(FancyArrowPatch((3.9, 1.72), (23.1, 1.72), arrowstyle="-|>", mutation_scale=16, color=TEAL, lw=2))
    ax.text(13.5, 1.89, "观测前缀 → encoder / predictor", ha="center", color=TEAL, weight="bold")
    ax.text(18.25, 0.16, "z_t = E(o_0, …, o_t)", ha="center", color=TEAL, family="monospace", weight="bold")

    ax = axes[1]
    ax.set_title("B. 训练时：未来标签只作为预测目标，反向塑造当前表征", loc="left")
    for i in range(4):
        box(ax, i, 0.72, 0.85, 0.65, f"ctx {i}", BLUE, alpha=0.18)
    box(ax, 4.4, 0.62, 29.1, 0.85, "future labels: t+1 … t+30（训练 shell，不是运行时输入）", "#FFF7ED", ORANGE, 0.9, "//")
    ax.add_patch(FancyArrowPatch((2.1, 1.6), (20.0, 1.6), connectionstyle="arc3,rad=-0.12", arrowstyle="-|>", mutation_scale=16, color=GOLD, lw=2))
    ax.text(11.1, 1.96, "forecast loss → 更新 encoder 参数", ha="center", color=GOLD, weight="bold")
    ax.text(
        18.9,
        0.25,
        "关键区分：训练目标可以包含未来标签；运行时依赖集合仍然只有 o_0…o_t。",
        ha="center",
        color="#334155",
    )
    fig.suptitle("因果边界：预测性正则 ≠ 未来输入泄漏", fontsize=18, fontweight="bold", y=1.02, color="#0F172A")
    fig.tight_layout(h_pad=1.6)
    save(fig, "01_causal_pipeline.png")


def chart_ablation() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    labels = ["forecast h=30", "same-frame masked\n(no future loss)"]
    vals = [0.049960, 0.180959]
    bars = ax.bar(labels, vals, color=[TEAL, ORANGE], width=0.58)
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.007, f"{value:.5f}", ha="center", va="bottom", weight="bold")
    ax.set_title("严格配对：Stage-A 目标的影响", loc="left")
    ax.set_ylabel("Held-out test tactile relative ↓")
    ax.set_ylim(0, 0.215)
    ax.grid(axis="y")
    ax.text(0.5, 0.187, "masked / forecast = 3.62×", ha="center", color=ORANGE, weight="bold")
    ax.text(0.0, -0.23, "同一 predictor、loss、seed、online target source", transform=ax.transAxes, fontsize=9, color=GRAY)

    ax = axes[1]
    labels = ["forecast h=1", "forecast h=30"]
    vals = [0.1860, 0.1983]
    bars = ax.bar(labels, vals, color=[BLUE, GOLD], width=0.58)
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.007, f"{value:.4f}", ha="center", va="bottom", weight="bold")
    ax.set_title("等预算 horizon 对照：h=30 不是自动更优", loc="left")
    ax.set_ylabel("Test tactile relative ↓")
    ax.set_ylim(0, 0.23)
    ax.grid(axis="y")
    ax.text(0.5, 0.197, "h=1 略优 6.2%", ha="center", color=BLUE, weight="bold")
    ax.text(0.0, -0.23, "相同 Stage-A/Stage-B step、数据与 seed", transform=ax.transAxes, fontsize=9, color=GRAY)

    fig.suptitle("消融结论：保留 forecast 目标，但不把收益错误归因给“未来 30 帧本身”", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    save(fig, "02_ablation_evidence.png")


def chart_stage_a(manifest: dict) -> None:
    val = manifest["stage_a_validation"]
    test = manifest["stage_a_test"]
    groups = ["Joint relative", "Future relative"]
    data = np.array(
        [
            [val["joint_relative"], test["joint_relative"]],
            [val["future_relative"], test["future_relative"]],
        ]
    )
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    x = np.arange(len(groups))
    width = 0.3
    b1 = ax.bar(x - width / 2, data[:, 0], width, color=BLUE, label="Complete validation")
    b2 = ax.bar(x + width / 2, data[:, 1], width, color=TEAL, label="Held-out test")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.00055, f"{bar.get_height():.6f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, groups)
    ax.set_ylabel("Relative error ↓")
    ax.set_ylim(0, 0.021)
    ax.set_title("正式 Stage A：step 8000 的完整评估", loc="left")
    ax.grid(axis="y")
    ax.legend(ncol=2, loc="upper left")
    fig.tight_layout()
    save(fig, "03_stage_a_results.png")


def chart_stage_b(manifest: dict) -> None:
    val = manifest["stage_b_full_validation"]
    test = manifest["stage_b_held_out_test"]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    groups = ["Tactile relative", "Visual relative"]
    data = np.array([[val["tactile_relative"], test["tactile_relative"]], [val["visual_relative"], test["visual_relative"]]])
    x = np.arange(2)
    width = 0.3
    b1 = ax.bar(x - width / 2, data[:, 0], width, color=BLUE, label="Validation")
    b2 = ax.bar(x + width / 2, data[:, 1], width, color=TEAL, label="Test")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.12, f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(0.008, 1.6)
    ax.set_xticks(x, groups)
    ax.set_ylabel("Relative error ↓（对数轴）")
    ax.set_title("Stage B：以触觉预测为主要优化目标", loc="left")
    ax.grid(axis="y", which="both")
    ax.legend(ncol=2)

    ax = axes[1]
    probe_names = ["Contact AUROC", "Slip AUROC"]
    x = np.arange(2)
    width = 0.3
    probe_val = [val["contact_probe_auroc"], val["slip_probe_auroc"]]
    probe_test = [test["contact_probe_auroc"], test["slip_probe_auroc"]]
    b1 = ax.bar(x - width / 2, probe_val, width, color=BLUE, label="Validation")
    b2 = ax.bar(x + width / 2, probe_test, width, color=TEAL, label="Test")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003, f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.5, color=ORANGE, lw=1.5, ls="--", label="Chance")
    ax.set_xticks(x, probe_names)
    ax.set_ylim(0.46, 0.61)
    ax.set_ylabel("AUROC（0.5 为随机）")
    ax.set_title("线性语义 probe：仍接近随机水平", loc="left")
    ax.grid(axis="y")
    ax.legend(ncol=3, loc="upper left")
    fig.suptitle("正式 Stage B：触觉 latent 预测是主目标，视觉与语义 probe 不是强项", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "04_stage_b_results.png")


def chart_horizons() -> None:
    d = json.loads((FINAL / "per_horizon_test_metrics.json").read_text())
    h_t = d["tactile"]["horizons"]
    rel_t = d["tactile"]["relative"]
    h_v = d["visual"]["horizons"]
    rel_v = d["visual"]["relative"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    ax = axes[0]
    ax.plot(h_t, rel_t, marker="o", ms=3.5, lw=2, color=TEAL)
    ax.fill_between(h_t, rel_t, color=TEAL, alpha=0.1)
    ax.scatter([1, 5, 30], [rel_t[0], rel_t[4], rel_t[-1]], color=ORANGE, zorder=3)
    for h, y, dy in [(1, rel_t[0], -0.0026), (5, rel_t[4], -0.0021), (30, rel_t[-1], 0.0012)]:
        ax.text(h, y + dy, f"h={h}: {y:.5f}", ha="left" if h < 30 else "right", color=ORANGE, fontsize=9, weight="bold")
    ax.set_title("触觉 latent relative，按未来 horizon", loc="left")
    ax.set_xlabel("Future horizon")
    ax.set_ylabel("Relative error ↓")
    ax.set_xticks(range(0, 31, 5))
    ax.grid()
    ax.set_ylim(0.005, 0.024)

    ax = axes[1]
    ax.plot(h_v, rel_v, marker="o", ms=4, lw=2, color=BLUE)
    ax.fill_between(h_v, rel_v, color=BLUE, alpha=0.1)
    for idx in [0, 2, -1]:
        ax.scatter(h_v[idx], rel_v[idx], color=ORANGE, zorder=3)
        ax.annotate(f"h={h_v[idx]}: {rel_v[idx]:.3f}", (h_v[idx], rel_v[idx]), xytext=(6, 10), textcoords="offset points", color=ORANGE, fontsize=9)
    ax.set_title("视觉 latent relative，按未来 horizon", loc="left")
    ax.set_xlabel("Future horizon")
    ax.set_ylabel("Relative error ↓")
    ax.set_xticks(range(2, 31, 4))
    ax.grid()
    ax.set_ylim(0.4, 0.75)

    fig.suptitle("误差随 horizon 增长但总体稳定，提升不是只由短期 copy-last 获得", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "05_per_horizon.png")


def chart_query_tokens(manifest: dict) -> None:
    q = manifest["query_token_eval"]
    coverage = q["query_coverage"]["future_tactile"]
    seg = json.loads((FINAL / "query_token_segment_coverage.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})

    ax = axes[0]
    labels = ["All 4", "Token 0", "Token 1", "Token 2", "Token 3"]
    values = [coverage["all_4_tokens"], coverage["token_0"], coverage["token_1"], coverage["token_2"], coverage["token_3"]]
    bars = ax.bar(labels, values, color=[TEAL, BLUE, BLUE, BLUE, BLUE], width=0.68)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.003, f"{value:.4f}", ha="center", va="bottom", fontsize=9, weight="bold")
    ax.set_ylim(0.82, 0.842)
    ax.set_ylabel("Ridge R² ↑")
    ax.set_title("Query token 覆盖全局未来触觉 latent", loc="left")
    ax.grid(axis="y")
    ax.text(0.0, -0.2, f"Zero-predictor R² = {coverage['zero_predictor_r2']:.1f}（只用于标示任务难度）", transform=ax.transAxes, fontsize=9, color=GRAY)

    ax = axes[1]
    labels = ["对角匹配\n(Token i → Segment i)", "非对角匹配\n(Token i → Segment j≠i)"]
    values = [seg["diagonal_mean"], seg["off_diagonal_mean"]]
    bars = ax.barh(labels, values, color=[TEAL, GOLD], height=0.5)
    for bar, value in zip(bars, values):
        ax.text(value + 0.002, bar.get_y() + bar.get_height() / 2, f"{value:.6f}", va="center", fontsize=10, weight="bold")
    ax.set_xlim(0.82, 0.84)
    ax.set_xlabel("Ridge R²")
    ax.set_title("四个 token 是全局摘要，而非独立时间段专家", loc="left")
    ax.grid(axis="x")
    ax.text(0.0, -0.2, f"Mean pair cosine={q['query_geometry']['mean_offdiag_cosine']:.5f}；token std={q['query_geometry']['token_std']:.4f}", transform=ax.transAxes, fontsize=9, color=GRAY)

    fig.suptitle("Query token：信息量高、未坍缩，但功能高度冗余", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    save(fig, "06_query_tokens.png")


def chart_raw_tradeoff(manifest: dict, prev_manifest: dict) -> None:
    formal = manifest["raw_common_readout"]["formal"]
    previous = manifest["raw_common_readout"]["previous_final"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
    model_names = ["Previous\nfinal", "Formal\nnew"]
    colors = [GRAY, TEAL]

    ax = axes[0]
    vals = [prev_manifest["held_out_test"]["tactile_relative"], manifest["stage_b_held_out_test"]["tactile_relative"]]
    bars = ax.bar(model_names, vals, color=colors, width=0.58)
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0004, f"{value:.5f}", ha="center", va="bottom", fontsize=9, weight="bold")
    ax.set_ylim(0, 0.031)
    ax.set_title("模型各自 latent relative", loc="left")
    ax.set_ylabel("越低越好")
    ax.grid(axis="y")
    ax.text(0.5, 0.028, "新模型低 40.7%", ha="center", color=TEAL, weight="bold")

    ax = axes[1]
    vals = [previous["raw_mse"] * 1e5, formal["raw_mse"] * 1e5]
    bars = ax.bar(model_names, vals, color=colors, width=0.58)
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.5f}e-5", ha="center", va="bottom", fontsize=9, weight="bold")
    ax.set_ylim(4.45, 4.57)
    ax.set_title("共同 7×7 raw tactile MSE", loc="left")
    ax.set_ylabel("Raw test MSE；越低越好")
    ax.grid(axis="y")
    ax.text(0.5, 4.555, "新模型高约 0.36%", ha="center", color=ORANGE, weight="bold")

    ax = axes[2]
    vals = [previous["raw_relative"], formal["raw_relative"]]
    bars = ax.bar(model_names, vals, color=colors, width=0.58)
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0015, f"{value:.5f}", ha="center", va="bottom", fontsize=9, weight="bold")
    ax.set_ylim(2.225, 2.255)
    ax.set_title("Raw MSE / raw copy", loc="left")
    ax.set_ylabel("越低越好")
    ax.grid(axis="y")
    ax.text(0.5, 2.249, "原始触觉空间基本持平", ha="center", color=GRAY, weight="bold")

    fig.suptitle("关键风险控制：latent 指标提升，但不能外推成 raw tactile 同比提升", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "07_raw_tradeoff.png")


def build_html() -> None:
    md = create_markdown(plugins=["table", "strikethrough"])
    source = (HERE / "feishu_report.md").read_text(encoding="utf-8")
    html_body = md(source)

    def embed(match: re.Match[str]) -> str:
        rel = match.group(1)
        data = base64.b64encode((HERE / rel).read_bytes()).decode("ascii")
        return f'src="data:image/png;base64,{data}"'

    html_body = re.sub(r'src="([^"]+\.png)"', embed, html_body)
    template = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>触觉世界模型训练复盘</title>
<style>
body {{ margin:0; background:#f3f6f9; color:#1f2937; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif; line-height:1.75; }}
.page {{ max-width:1040px; margin:28px auto 60px; background:#fff; padding:54px 64px 72px; box-shadow:0 12px 40px rgba(15,23,42,.08); border-radius:14px; }}
h1 {{ font-size:34px; line-height:1.3; color:#0f172a; margin:0 0 8px; }}
h2 {{ font-size:24px; color:#123b5d; border-left:5px solid #2a9d8f; padding-left:12px; margin-top:38px; }}
h3 {{ font-size:18px; color:#244b69; margin-top:26px; }}
p, li {{ font-size:15.5px; }}
table {{ border-collapse:collapse; width:100%; margin:16px 0 22px; font-size:14px; }}
th {{ background:#eaf3f7; color:#173b57; text-align:left; }}
th,td {{ border:1px solid #dbe4ea; padding:9px 12px; vertical-align:top; }}
tr:nth-child(even) td {{ background:#f8fbfc; }}
img {{ display:block; width:100%; height:auto; margin:18px auto 10px; border:1px solid #e5edf2; border-radius:10px; }}
blockquote {{ margin:18px 0; padding:14px 18px; background:#f0f8f6; border-left:4px solid #2a9d8f; color:#234; }}
code {{ background:#edf2f7; padding:2px 6px; border-radius:5px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
pre {{ background:#102a43; color:#e6f1ff; padding:18px 20px; border-radius:10px; overflow:auto; }}
pre code {{ background:transparent; padding:0; color:inherit; }}
.kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:26px 0 32px; }}
.kpi {{ background:linear-gradient(155deg,#f2f8fc,#e9f6f3); border:1px solid #d9e9ef; border-radius:12px; padding:16px 18px; }}
.kpi b {{ display:block; font-size:25px; color:#176b73; line-height:1.2; margin-bottom:4px; }}
.kpi span {{ font-size:13px; color:#52616b; }}
@media (max-width:760px) {{ .page {{ padding:28px 20px; margin:0; border-radius:0; }} .kpis {{ grid-template-columns:repeat(2,1fr); }} h1 {{ font-size:28px; }} }}
</style>
</head>
<body>
<main class="page">
<div class="kpis">
  <div class="kpi"><b>0.015516</b><span>Stage B 测试触觉 relative</span></div>
  <div class="kpi"><b>0.8362</b><span>Query token 全局 R²</span></div>
  <div class="kpi"><b>4 frames</b><span>运行时观测历史</span></div>
  <div class="kpi"><b>No action</b><span>Stage B 不输出动作</span></div>
</div>
{html_body}
</main>
</body>
</html>
"""
    (HERE / "feishu_copy_source.html").write_text(template, encoding="utf-8")


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    configure_style()
    manifest = json.loads((FINAL / "model_manifest.json").read_text())
    prev_manifest = json.loads((PREV / "model_manifest.json").read_text())
    chart_causal_pipeline()
    chart_ablation()
    chart_stage_a(manifest)
    chart_stage_b(manifest)
    chart_horizons()
    chart_query_tokens(manifest)
    chart_raw_tradeoff(manifest, prev_manifest)
    build_html()
    print(f"Generated {len(list(IMAGES.glob('*.png')))} charts")
    print(HERE / "feishu_copy_source.html")


if __name__ == "__main__":
    main()
