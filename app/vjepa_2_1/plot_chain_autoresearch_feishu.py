"""Generate presentation-grade figures for the Feishu autoresearch report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/runs/autoresearch_20261112"
OUT = ROOT / "docs/assets/tactile_chain_autoresearch_feishu"

INK = "#14213D"
MUTED = "#667085"
GRID = "#E7ECF3"
BLUE = "#2563EB"
CYAN = "#0891B2"
ORANGE = "#EA580C"
GREEN = "#16A34A"
PURPLE = "#7C3AED"
RED = "#DC2626"
SLATE = "#64748B"

FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
if FONT_PATH.is_file():
    font_manager.fontManager.addfont(str(FONT_PATH))
    CJK_FONT = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
else:
    CJK_FONT = "DejaVu Sans"

plt.rcParams.update(
    {
        "font.family": CJK_FONT,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CBD5E1",
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
    }
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT / name,
        dpi=220,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def clean_axes(ax: plt.Axes, *, ygrid: bool = True) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CBD5E1")
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=10)


def value_labels(ax: plt.Axes, bars, fmt: str = "{:.4f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=INK,
        )


def rounded_note(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    color: str = "#EFF6FF",
    edge: str = "#BFDBFE",
    fontsize: int = 10,
) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        bbox={
            "boxstyle": "round,pad=0.55,rounding_size=0.6",
            "facecolor": color,
            "edgecolor": edge,
            "linewidth": 1.2,
        },
    )


def plot_attention() -> None:
    labels = ["因果时间 attention\nper_frame_causal", "全 clip 双向 attention\nfull_clip_bidirectional"]
    values = [0.74983, 0.87152]
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    bars = ax.bar(labels, values, color=[GREEN, RED], width=0.56)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("validation score（越低越好）")
    ax.set_title(
        "相同 4→30 预算：因果时间结构明显领先",
        fontsize=17,
        fontweight="bold",
        pad=16,
    )
    value_labels(ax, bars)
    clean_axes(ax)
    ax.annotate(
        "score 降低 14.0%",
        xy=(0.5, 0.86),
        xycoords="axes fraction",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color=GREEN,
    )
    rounded_note(
        ax,
        0.5,
        0.12,
        "双向版本容易长时间停在 copy-last，未来 latent 缺乏可预测变化",
        color="#FFF7ED",
        edge="#FED7AA",
    )
    save(fig, "fig_01_attention.png")


def plot_horizon() -> None:
    horizons = ["h1", "h5", "h10", "h30"]
    stage_a = [0.14218, 0.06534, 0.05553, 0.33934]
    downstream = [np.nan, 0.24446, np.nan, 0.03638]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), gridspec_kw={"wspace": 0.26})

    ax = axes[0]
    colors = [ORANGE, CYAN, BLUE, RED]
    bars = ax.bar(horizons, stage_a, color=colors, width=0.6)
    value_labels(ax, bars, "{:.3f}")
    ax.set_ylim(0, 0.39)
    ax.set_ylabel("Stage A 1000-step validation")
    ax.set_title("Stage A 自身指标：短 horizon 看似更好", fontsize=14, fontweight="bold")
    clean_axes(ax)
    rounded_note(
        ax,
        0.5,
        0.84,
        "注意：不能据此选型",
        color="#FEF2F2",
        edge="#FECACA",
        fontsize=11,
    )

    ax = axes[1]
    labels = ["h5", "h30"]
    values = [downstream[1], downstream[3]]
    bars = ax.bar(labels, values, color=[ORANGE, GREEN], width=0.48)
    ax.set_yscale("log")
    ax.set_ylim(0.025, 0.35)
    ax.set_ylabel("Stage B full-test tactile relative（log）")
    ax.set_title("冻结 encoder 后：只有 h30 真正可用", fontsize=14, fontweight="bold")
    value_labels(ax, bars, "{:.4f}")
    clean_axes(ax)
    rounded_note(
        ax,
        0.5,
        0.82,
        "h30 误差降低 85.1%",
        color="#ECFDF5",
        edge="#A7F3D0",
        fontsize=11,
    )

    fig.suptitle(
        "连续 horizon 仲裁：Stage A loss 会“骗人”，下游指标决定最终配置",
        fontsize=17,
        fontweight="bold",
        y=1.02,
    )
    save(fig, "fig_02_horizon.png")


def plot_mask_grid() -> None:
    labels = ["整帧 mask\n7×7", "部分 token\n7×7", "整帧 mask\n21×14", "部分 token\n21×14"]
    values = [0.0274, 0.0889, 0.0487, 0.1160]
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    bars = ax.bar(labels, values, color=[GREEN, ORANGE, BLUE, RED], width=0.62)
    ax.set_yscale("log")
    ax.set_ylim(0.02, 0.16)
    ax.set_ylabel("best validation joint relative（log）")
    ax.set_title(
        "空间粒度与 mask 粒度：7×7 + 整帧未来 mask 最优",
        fontsize=17,
        fontweight="bold",
        pad=16,
    )
    value_labels(ax, bars, "{:.4f}")
    clean_axes(ax)
    rounded_note(
        ax,
        0.5,
        0.80,
        "保留逐帧可见 token 会形成 teacher-forcing 捷径",
        color="#FFF7ED",
        edge="#FED7AA",
    )
    save(fig, "fig_03_mask_grid.png")


def plot_stage_a_curve() -> None:
    data = load_json(BASE / "stage_a/final7750_horizon_metrics.json")["test"]
    offsets = np.asarray(data["horizon_offsets"])
    relative = np.asarray(data["relative"])
    cosine = np.asarray(data["cosine"])
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), gridspec_kw={"wspace": 0.28})

    ax = axes[0]
    ax.plot(offsets, relative, color=BLUE, linewidth=2.4, marker="o", markersize=4)
    ax.fill_between(offsets, relative, color=BLUE, alpha=0.08)
    ax.set_xlabel("future offset (frames)")
    ax.set_ylabel("relative latent error")
    ax.set_title("最终 Stage A：future-offset 曲线", fontsize=14, fontweight="bold")
    clean_axes(ax)
    ax.text(offsets[0], relative[0], f"  t+1 {relative[0]:.4f}", va="top", color=BLUE)
    ax.text(offsets[-1], relative[-1], f"  t+30 {relative[-1]:.4f}", va="bottom", ha="right", color=BLUE)

    ax = axes[1]
    cosine_gap = (1.0 - cosine) * 1e6
    ax.plot(
        offsets,
        cosine_gap,
        color=GREEN,
        linewidth=2.4,
        marker="o",
        markersize=4,
    )
    ax.set_ylim(0, 10)
    ax.set_xlabel("future offset (frames)")
    ax.set_ylabel("1 - cosine similarity (ppm)")
    ax.set_title("方向误差始终低于 10 ppm", fontsize=14, fontweight="bold")
    clean_axes(ax)
    ax.axhline(0, color=SLATE, linestyle="--", linewidth=1)

    fig.suptitle(
        "最终 Stage A：aggregate relative = 0.03408",
        fontsize=17,
        fontweight="bold",
        y=1.03,
    )
    save(fig, "fig_04_stage_a_curve.png")


def plot_projection_query() -> None:
    labels = ["MLP\n4 tokens", "normed-MLP\n4 tokens", "MLP\n1 token"]
    values = [0.247135, 0.232865, 0.232758]
    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    bars = ax.bar(labels, values, color=[SLATE, BLUE, PURPLE], width=0.58)
    ax.set_ylim(0.20, 0.26)
    ax.set_ylabel("750-step validation selection")
    ax.set_title(
        "短程消融：投影归一化与单 query 接近，但 no-drop 下不复现",
        fontsize=16,
        fontweight="bold",
        pad=16,
    )
    value_labels(ax, bars)
    clean_axes(ax)
    rounded_note(
        ax,
        0.5,
        0.14,
        "最终保留：普通 MLP + 4 query tokens",
        color="#ECFDF5",
        edge="#A7F3D0",
        fontsize=11,
    )
    save(fig, "fig_05_projection_query.png")


def plot_dropout() -> None:
    seeds = ["seed 0", "seed 1", "seed 2"]
    dropout = [0.258264, 0.222094, 0.196565]
    no_dropout = [0.202693, 0.203082, 0.198743]
    x = np.arange(len(seeds))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    bars1 = ax.bar(x - width / 2, dropout, width, label="dropout = 0.1", color=ORANGE)
    bars2 = ax.bar(x + width / 2, no_dropout, width, label="dropout = 0", color=GREEN)
    ax.set_xticks(x, seeds)
    ax.set_ylim(0.18, 0.27)
    ax.set_ylabel("best validation selection")
    ax.set_title(
        "随机 chain/arm dropout 三 seed 对照：真实 contact mask 已足够",
        fontsize=16,
        fontweight="bold",
        pad=16,
    )
    value_labels(ax, bars1, "{:.3f}")
    value_labels(ax, bars2, "{:.3f}")
    clean_axes(ax)
    ax.legend(frameon=False, loc="upper right")
    ax.axhline(np.mean(no_dropout), color=GREEN, linestyle="--", linewidth=1.4)
    ax.text(
        2.42,
        np.mean(no_dropout) + 0.002,
        f"no-drop mean {np.mean(no_dropout):.4f}",
        ha="right",
        color=GREEN,
        fontsize=10,
        fontweight="bold",
    )
    save(fig, "fig_06_dropout.png")


def plot_test_stability() -> None:
    labels = ["seed 0", "seed 1", "seed 2", "mean"]
    values = [0.036253, 0.034756, 0.037120, 0.036043]
    baseline = 0.05273
    fig, ax = plt.subplots(figsize=(10.2, 5.1))
    bars = ax.bar(labels, values, color=[BLUE, GREEN, CYAN, PURPLE], width=0.56)
    ax.axhline(baseline, color=RED, linestyle="--", linewidth=1.5)
    ax.text(3.45, baseline + 0.001, "旧 baseline 0.05273", ha="right", color=RED, fontsize=10)
    ax.set_ylim(0, 0.06)
    ax.set_ylabel("full-test tactile relative")
    ax.set_title(
        "最终 no-drop 模型：三 seed held-out test 稳定且显著优于 baseline",
        fontsize=16,
        fontweight="bold",
        pad=16,
    )
    value_labels(ax, bars, "{:.5f}")
    clean_axes(ax)
    rounded_note(
        ax,
        0.68,
        0.74,
        "相对旧 baseline 降低 31.6%",
        color="#ECFDF5",
        edge="#A7F3D0",
        fontsize=11,
    )
    save(fig, "fig_07_test_stability.png")


def plot_query_tokens() -> None:
    probe = load_json(BASE / "stage_b/final_nodrop_seed1_query_probe.json")
    coverage = probe["query_coverage"]
    matrix = np.asarray(
        [
            [coverage["history_tactile"][f"token_{index}"] for index in range(4)],
            [coverage["future_tactile"][f"token_{index}"] for index in range(4)],
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), gridspec_kw={"wspace": 0.35})

    ax = axes[0]
    image = ax.imshow(matrix, cmap="Blues", vmin=0.82, vmax=0.85, aspect="auto")
    ax.set_xticks(range(4), [f"Q{i}" for i in range(4)])
    ax.set_yticks(range(2), ["History tactile", "Future tactile"])
    for row in range(2):
        for column in range(4):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:.4f}",
                ha="center",
                va="center",
                color=INK,
                fontsize=10,
                fontweight="bold",
            )
    ax.set_title("独立 ridge probe：每个 token 都覆盖历史/未来", fontsize=13, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.05, pad=0.04, label="held-out R²")

    ax = axes[1]
    labels = ["history\nall 4 tokens", "future\nall 4 tokens", "mean |cos|", "effective rank / 10"]
    values = [
        coverage["history_tactile"]["all_4_tokens"],
        coverage["future_tactile"]["all_4_tokens"],
        probe["query_geometry"]["mean_offdiag_cosine"],
        probe["query_geometry"]["effective_rank"] / 10,
    ]
    colors = [BLUE, CYAN, GREEN, PURPLE]
    bars = ax.barh(labels, values, color=colors)
    ax.set_xlim(0, 1)
    for bar, raw in zip(bars, [
        coverage["history_tactile"]["all_4_tokens"],
        coverage["future_tactile"]["all_4_tokens"],
        probe["query_geometry"]["mean_offdiag_cosine"],
        probe["query_geometry"]["effective_rank"],
    ]):
        label = f"{raw:.4f}" if raw < 1 else f"{raw:.2f}"
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            color=INK,
            fontsize=10,
            fontweight="bold",
        )
    clean_axes(ax, ygrid=False)
    ax.grid(axis="x", color=GRID, linewidth=0.9)
    ax.set_title("token 未塌缩，但不是强逐槽分工", fontsize=13, fontweight="bold")

    fig.suptitle(
        "4 个 query token：信息覆盖稳定，更多 token 不等于更好",
        fontsize=17,
        fontweight="bold",
        y=1.03,
    )
    save(fig, "fig_08_query_tokens.png")


def main() -> None:
    plot_attention()
    plot_horizon()
    plot_mask_grid()
    plot_stage_a_curve()
    plot_projection_query()
    plot_dropout()
    plot_test_stability()
    plot_query_tokens()
    print(OUT)


if __name__ == "__main__":
    main()
