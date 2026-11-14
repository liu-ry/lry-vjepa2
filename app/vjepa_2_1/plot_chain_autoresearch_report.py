"""Generate figures for the tactile-chain autoresearch report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data/runs"
OUT = ROOT / "docs/assets/tactile_chain_autoresearch"

INK = "#172033"
MUTED = "#64748B"
GRID = "#E2E8F0"
BLUE = "#2563EB"
CYAN = "#0891B2"
ORANGE = "#EA580C"
GREEN = "#16A34A"
PURPLE = "#7C3AED"
RED = "#DC2626"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def style(ax):
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#B8C4D6")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT / name,
        bbox_inches="tight",
        dpi=210,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def convergence(ax, run: Path, label: str, color: str):
    metrics = load_json(run / "metrics.json")
    records = metrics.get("records", {})
    points = sorted(
        (
            int(step),
            value.get("validation", {}).get("joint_relative"),
        )
        for step, value in records.items()
        if value.get("validation", {}).get("joint_relative") is not None
    )
    if points:
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            label=label,
            color=color,
            linewidth=2,
        )


def plot_stage_a():
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    convergence(
        ax,
        RUNS / "chain_autoresearch_stage_a_final/causal_7x7_future_frames",
        "Forecast-30 · causal · 7×7",
        BLUE,
    )
    convergence(
        ax,
        RUNS / "chain_autoresearch_stage_a_final/causal_21x14_future_frames",
        "Forecast-30 · causal · 21×14",
        CYAN,
    )
    convergence(
        ax,
        RUNS / "chain_autoresearch_stage_a_clipmask/bidir_21x14",
        "V-JEPA clip mask · bidir · 21×14",
        PURPLE,
    )
    sparse = RUNS / "chain_autoresearch_stage_a_sparse/causal_21x14"
    if (sparse / "metrics.json").is_file():
        convergence(
            ax,
            sparse,
            "Sparse horizons · causal · 21×14",
            ORANGE,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Validation relative latent error (log scale)")
    ax.set_title("Stage A objectives and temporal structure", color=INK, weight="bold")
    style(ax)
    ax.legend(frameon=False, ncols=2)
    save(fig, "stage_a_convergence.png")


def plot_stage_a_bar():
    candidates = [
        (
            "Forecast-30\ncausal 7×7",
            RUNS / "chain_autoresearch_stage_a_final/causal_7x7_future_frames",
        ),
        (
            "Forecast-30\ncausal 21×14",
            RUNS / "chain_autoresearch_stage_a_final/causal_21x14_future_frames",
        ),
        (
            "Clip mask\nbidir 21×14",
            RUNS / "chain_autoresearch_stage_a_clipmask/bidir_21x14",
        ),
        (
            "Sparse horizon\ncausal 21×14",
            RUNS / "chain_autoresearch_stage_a_sparse/causal_21x14",
        ),
    ]
    labels, values, colors = [], [], []
    for index, (label, run) in enumerate(candidates):
        if not (run / "metrics.json").is_file():
            continue
        metrics = load_json(run / "metrics.json")
        record = metrics.get("records", {}).get(str(metrics["best_step"]), {})
        value = record.get("validation", {}).get("joint_relative")
        if value is None:
            continue
        labels.append(label)
        values.append(value)
        colors.append((BLUE, CYAN, PURPLE, ORANGE)[index])
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor="white")
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.bar_label(bars, fmt="%.3f", padding=3, color=INK, fontsize=9)
    ax.set_ylabel("Best validation relative error")
    ax.set_title("Stage A objective comparison", color=INK, weight="bold")
    style(ax)
    save(fig, "stage_a_comparison.png")


def plot_attention():
    labels = ["Causal\nfull-clip mask", "Bidirectional\nfull-clip mask"]
    runs = [
        RUNS / "chain_autoresearch_stage_a_final_causal_7x7_future_frames.log",
        RUNS / "chain_autoresearch_stage_a_final_bidir_7x7_future_frames.log",
    ]
    values = [0.0274, 0.9132]
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
    bars = ax.bar(labels, values, color=[GREEN, RED], width=0.52)
    ax.bar_label(bars, fmt="%.3f", padding=4, color=INK)
    ax.set_ylabel("Best validation joint relative error")
    ax.set_title("Fixed 4→30 forecast: temporal attention ablation",
                 color=INK, weight="bold")
    style(ax)
    save(fig, "attention_ablation.png")


def plot_mask_grid():
    labels = ["Full frame\n7×7", "Partial token\n7×7",
              "Full frame\n21×14", "Partial token\n21×14"]
    values = [0.0274, 0.0889, 0.0487, 0.1160]
    fig, ax = plt.subplots(figsize=(8.5, 4.5), facecolor="white")
    bars = ax.bar(labels, values, color=[GREEN, ORANGE, BLUE, RED], width=0.62)
    ax.bar_label(bars, fmt="%.3f", padding=3, color=INK)
    ax.set_yscale("log")
    ax.set_ylabel("Best validation joint relative error")
    ax.set_title("Mask granularity and target resolution", color=INK, weight="bold")
    style(ax)
    save(fig, "mask_grid_ablation.png")


def plot_data_mix():
    labels = ["Four chains", "Left arm only", "Right arm only"]
    values = [3150, 1309, 1777]
    fig, ax = plt.subplots(figsize=(7, 4.4), facecolor="white")
    bars = ax.bar(labels, values, color=[BLUE, GREEN, ORANGE], width=0.56)
    ax.bar_label(bars, padding=3, color=INK)
    ax.set_ylabel("Episodes")
    ax.set_title("Confirmed four-chain episode composition", color=INK, weight="bold")
    style(ax)
    save(fig, "data_mix.png")


def plot_stage_b():
    early_segments = [
        RUNS / "chain_autoresearch_final_pf1/stageb_forecast30_7x7_d3",
        RUNS / "chain_autoresearch_final_resume/stageb_forecast30_7x7_d3",
    ]
    balanced_segment = (
        RUNS / "chain_autoresearch_balanced2/stageb_forecast30_7x7_d3"
    )
    if not all(
        (segment / "metrics.json").is_file()
        for segment in early_segments + [balanced_segment]
    ):
        return
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    def points_for(runs):
        points = []
        for run in runs:
            metrics = load_json(run / "metrics.json")
            records = metrics.get("records", {})
            points.extend(
                sorted(
                    (int(step), value["validation"].get("tactile_relative"))
                    for step, value in records.items()
                    if value.get("validation", {}).get("tactile_relative")
                    is not None
                )
            )
        return sorted(points)

    early = points_for(early_segments)
    balanced = points_for([balanced_segment])
    ax.plot(
        [point[0] for point in early],
        [point[1] for point in early],
        label="Natural contact-window sampling",
        color=ORANGE,
        linewidth=2.2,
        marker="o",
        markersize=3,
    )
    ax.plot(
        [point[0] for point in balanced],
        [point[1] for point in balanced],
        label="Contact-balanced validation",
        color=BLUE,
        linewidth=2.2,
        marker="o",
        markersize=3,
    )
    ax.axvline(1700, color=MUTED, linestyle="--", linewidth=1)
    ax.text(1710, max(point[1] for point in balanced),
            " balanced\n validation", color=MUTED, fontsize=8, va="top")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Validation tactile relative error")
    ax.set_title("Final Stage B convergence", color=INK, weight="bold")
    style(ax)
    ax.legend(frameon=False, fontsize=8)
    save(fig, "stage_b_convergence.png")


def plot_stage_b_comparison():
    candidates = [
        ("Forecast-30\n7×7 d2", "stageb_forecast30_7x7_d2"),
        ("Forecast-30\n7×7 d3", "stageb_forecast30_7x7_d3"),
        ("Forecast-30\n21×14 d2", "stageb_forecast30_21x14_d2"),
        ("Sparse horizon\n21×14", "stageb_sparse_causal21_d2"),
        ("Clip mask\n21×14", "stageb_clipmask_bidir21_d2"),
    ]
    labels, values = [], []
    for label, name in candidates:
        path = RUNS / "chain_autoresearch_final_pf1" / name / "metrics.json"
        if not path.is_file():
            continue
        metrics = load_json(path)
        record = metrics.get("records", {}).get(str(metrics["best_step"]), {})
        value = record.get("validation", {}).get("tactile_relative")
        if value is None:
            continue
        labels.append(label)
        values.append(value)
    fig, ax = plt.subplots(figsize=(9, 4.7), facecolor="white")
    bars = ax.bar(
        labels,
        values,
        color=[BLUE, ORANGE, CYAN, GREEN, PURPLE],
        width=0.6,
    )
    ax.bar_label(bars, fmt="%.3f", padding=3, color=INK)
    ax.set_yscale("log")
    ax.set_ylabel("Validation tactile relative error")
    ax.set_title("Stage B objective comparison at equal data budget",
                 color=INK, weight="bold")
    style(ax)
    save(fig, "stage_b_comparison.png")


def plot_query_coverage():
    candidates = sorted(OUT.parent.glob("tactile_chain_autoresearch_query*.json"))
    if not candidates:
        return
    report = load_json(candidates[0])
    coverage = report["query_coverage"]
    rows, columns, values = [], ["history", "future"], []
    for target in ("history_tactile", "future_tactile"):
        for token in range(4):
            rows.append(f"{target.split('_')[0]} · token {token}")
    for target in ("history_tactile", "future_tactile"):
        values.append(
            [coverage[target][f"token_{token}"] for token in range(4)]
        )
    matrix = np.asarray(values)
    fig, ax = plt.subplots(figsize=(7.5, 3.6), facecolor="white")
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), [f"Q{index}" for index in range(4)])
    ax.set_yticks(range(2), ["History tactile", "Future tactile"])
    for row in range(2):
        for column in range(4):
            ax.text(column, row, f"{matrix[row, column]:.3f}",
                    ha="center", va="center", color=INK, fontsize=9)
    ax.set_title("Linear information coverage of each query token",
                 color=INK, weight="bold")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03, label="Held-out R²")
    save(fig, "query_token_coverage.png")


def plot_architecture():
    fig, ax = plt.subplots(figsize=(13, 5.2), facecolor="white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        rect = plt.Rectangle(
            (x - w / 2, y - h / 2), w, h,
            facecolor=color, edgecolor="#94A3B8", linewidth=1.2,
            joinstyle="round",
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", color=INK, fontsize=9)

    box(10, 39, 14, 12, "4 tactile chains\n(left L/R, right L/R)", "#DBEAFE")
    box(29, 39, 13, 12, "Shared\nTactileEncoder", "#EDE9FE")
    box(47, 39, 14, 12, "Chain-local\nlatent tokens", "#CFFAFE")
    box(68, 39, 16, 12, "Stage B\ncross-chain predictor", "#FEF3C7")
    box(90, 43, 13, 10, "Future tactile\nlatent grid", "#DCFCE7")
    box(90, 29, 13, 10, "4 condition\ntokens", "#FCE7F3")

    box(10, 13, 14, 10, "2 RGB streams", "#DBEAFE")
    box(29, 13, 13, 10, "Frozen\nV-JEPA 2.1", "#EDE9FE")
    box(47, 13, 14, 10, "Arm visual\nlatents", "#CFFAFE")
    box(68, 13, 16, 10, "2 state streams\n+ contact masks", "#F1F5F9")

    for start, end in (
        ((17, 39), (22.5, 39)),
        ((35.5, 39), (40, 39)),
        ((54, 39), (60, 39)),
        ((76, 39), (83.5, 43)),
        ((76, 39), (83.5, 29)),
        ((17, 13), (22.5, 13)),
        ((35.5, 13), (40, 13)),
        ((54, 13), (60, 13)),
        ((68, 18), (68, 33)),
    ):
        ax.annotate(
            "", xy=end, xytext=start,
            arrowprops={"arrowstyle": "-|>", "color": "#64748B",
                        "lw": 1.5},
        )
    ax.text(
        50, 4,
        "Stage A trains each chain independently. Stage B fuses chains only "
        "after four chain-local encodings.",
        ha="center", color=MUTED, fontsize=9, style="italic",
    )
    save(fig, "architecture.png")


def main():
    plot_architecture()
    plot_data_mix()
    plot_stage_a()
    plot_stage_a_bar()
    plot_attention()
    plot_mask_grid()
    plot_stage_b()
    plot_stage_b_comparison()
    plot_query_coverage()
    print(OUT)


if __name__ == "__main__":
    main()
