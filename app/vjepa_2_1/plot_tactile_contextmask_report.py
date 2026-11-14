"""Generate publication-ready figures for the tactile context-mask report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data/runs/tactile_contextmask_best"
OUT = ROOT / "docs/assets/tactile_contextmask"

INK = "#152238"
MUTED = "#607089"
GRID = "#DDE4EE"
BLUE = "#2563EB"
CYAN = "#0891B2"
ORANGE = "#EA580C"
GREEN = "#16A34A"
PURPLE = "#7C3AED"
BG = "#F7F9FC"


def _load(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _style(ax):
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#B8C4D6")
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def _save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT / name,
        format="svg",
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        metadata={"Date": None, "Creator": "matplotlib"},
    )
    fig.savefig(
        OUT / f"{Path(name).stem}.png",
        format="png",
        dpi=220,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def _diagram_box(ax, x, y, width, height, text, color, edge, fontsize=10):
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.25,rounding_size=0.8",
        facecolor=color,
        edgecolor=edge,
        linewidth=1.3,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        color=INK,
        fontsize=fontsize,
        linespacing=1.25,
        fontweight="bold",
    )


def _diagram_arrow(
    ax, start, end, color=MUTED, dashed=False, connectionstyle=None
):
    arrowprops = {
        "arrowstyle": "-|>",
        "color": color,
        "lw": 1.6,
        "linestyle": "--" if dashed else "-",
        "shrinkA": 5,
        "shrinkB": 5,
    }
    if connectionstyle:
        arrowprops["connectionstyle"] = connectionstyle
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=arrowprops,
    )


def plot_architecture_overview():
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), facecolor=BG)
    labels = (
        "Stage A · Masked tactile pretraining",
        "Stage B · Stable forecasting + query readout",
    )
    for panel, (ax, label) in enumerate(zip(axes, labels)):
        ax.set_xlim(0, 102)
        ax.set_ylim(0, 100)
        ax.axis("off")
        ax.text(
            1,
            96,
            label,
            ha="left",
            va="top",
            color=INK,
            fontsize=14,
            fontweight="bold",
        )
        if panel == 0:
            _diagram_box(
                ax,
                6,
                61,
                10,
                15,
                "Left GelSight\n+ Right GelSight",
                "#E0F2FE",
                BLUE,
                9,
            )
            _diagram_box(
                ax, 21, 61, 11, 15, "Shared\ntactile ViT", "#EDE9FE", PURPLE
            )
            _diagram_box(
                ax,
                37,
                61,
                12,
                15,
                "Future spatial\ntokens",
                "#F0F9FF",
                CYAN,
            )
            _diagram_box(
                ax,
                53,
                61,
                12,
                15,
                "Mask 75%\nper future frame",
                "#FEF3C7",
                ORANGE,
            )
            _diagram_box(
                ax,
                69,
                61,
                12,
                15,
                "Masked tactile\npredictor",
                "#EDE9FE",
                PURPLE,
            )
            _diagram_box(
                ax, 87, 73, 13, 12, "Context\ntokens", "#DCFCE7", GREEN
            )
            _diagram_box(
                ax,
                87,
                49,
                13,
                12,
                "Future tactile\nlatents · 30",
                "#CFFAFE",
                CYAN,
            )
            _diagram_box(
                ax,
                69,
                23,
                13,
                12,
                "EMA target\nstop-grad",
                "#E2E8F0",
                MUTED,
                9,
            )
            _diagram_box(
                ax,
                87,
                23,
                13,
                12,
                "Context + future\nrelative losses",
                "#FEE2E2",
                "#DC2626",
                9,
            )
            for start, end in (
                ((11, 61), (15.5, 61)),
                ((26.5, 61), (31, 61)),
                ((43, 61), (47, 61)),
                ((59, 61), (63, 61)),
                ((75, 61), (80.5, 73)),
                ((75, 61), (80.5, 49)),
            ):
                _diagram_arrow(ax, start, end)
            _diagram_arrow(
                ax,
                (87, 67),
                (87, 29),
                color="#DC2626",
                connectionstyle="arc3,rad=-0.45",
            )
            _diagram_arrow(ax, (87, 43), (87, 29), color="#DC2626")
            _diagram_arrow(ax, (75.5, 23), (80.5, 23), color="#DC2626")
            ax.text(
                50,
                35,
                "No future action input: passive latent forecasting",
                ha="center",
                color=MUTED,
                fontsize=10,
                style="italic",
            )
        else:
            _diagram_box(
                ax, 6, 73, 10, 10, "4 × touch\nhistory", "#CFFAFE", CYAN, 9
            )
            _diagram_box(
                ax,
                6,
                56,
                10,
                10,
                "4 × V-JEPA\nvisual history",
                "#DBEAFE",
                BLUE,
                9,
            )
            _diagram_box(
                ax,
                6,
                39,
                10,
                10,
                "4 × robot\nstate history",
                "#FEF3C7",
                ORANGE,
                9,
            )
            _diagram_box(
                ax,
                23,
                56,
                12,
                20,
                "Dense latent\npredictor",
                "#EDE9FE",
                PURPLE,
            )
            _diagram_box(
                ax,
                42,
                67,
                13,
                11,
                "Future tactile\nlatent · 30",
                "#CFFAFE",
                CYAN,
                9,
            )
            _diagram_box(
                ax,
                42,
                47,
                13,
                11,
                "Future visual\n15 waypoints",
                "#DBEAFE",
                BLUE,
                9,
            )
            _diagram_box(
                ax,
                60,
                57,
                12,
                24,
                "Condition\nmemory",
                "#DCFCE7",
                GREEN,
            )
            _diagram_box(
                ax,
                60,
                22,
                12,
                10,
                "4 learnable\nqueries",
                "#F3E8FF",
                PURPLE,
                9,
            )
            _diagram_box(
                ax,
                79,
                57,
                14,
                20,
                "Cross-attention\nreadout",
                "#EDE9FE",
                PURPLE,
            )
            _diagram_box(
                ax,
                95,
                57,
                8,
                14,
                "4 condition\ntokens",
                "#DCFCE7",
                GREEN,
                9,
            )
            for start, end in (
                ((11, 73), (17, 66)),
                ((11, 56), (17, 56)),
                ((11, 39), (17, 46)),
                ((29, 62), (35.5, 67)),
                ((29, 50), (35.5, 47)),
                ((48.5, 67), (54, 62)),
                ((48.5, 47), (54, 52)),
                ((66, 27), (66, 45)),
            ):
                _diagram_arrow(ax, start, end)
            _diagram_arrow(ax, (66, 72), (72, 67))
            _diagram_arrow(ax, (72, 57), (86, 57))
            ax.text(
                47,
                8,
                "Queries summarize history, predicted future, and their trajectory",
                ha="center",
                color=MUTED,
                fontsize=10,
                style="italic",
            )
    fig.suptitle(
        "Two-stage tactile world-model architecture",
        x=0.06,
        ha="left",
        color=INK,
        fontsize=18,
        fontweight="bold",
    )
    _save(fig, "architecture_overview.svg")


def plot_decision_path():
    fig, ax = plt.subplots(figsize=(16, 7.5), facecolor=BG)
    ax.set_xlim(0, 103)
    ax.set_ylim(0, 100)
    ax.axis("off")
    adopted = (
        (8, "Baseline\nfuture-only"),
        (25, "Context 0.1\nPredict 1.0"),
        (42, "2×2 coarse\nlatent target"),
        (59, "Spatial-token\nmask 75%"),
        (76, "Warm start +\nfrozen latent"),
        (94, "Test relative\n0.2288"),
    )
    rejected = (
        (8, "Absolute latent\ncopy-last risk"),
        (25, "Equal weights\nTemporal collapse"),
        (42, "4×4 target\ngrid too hard"),
        (59, "future_frames control\nTest 0.3540"),
        (76, "Random 0.6391\nEncoder tuning 0.4406"),
    )
    for index, (x, text) in enumerate(adopted):
        color = "#DCFCE7" if index == len(adopted) - 1 else "#E0F2FE"
        edge = GREEN if index == len(adopted) - 1 else BLUE
        _diagram_box(ax, x, 66, 14, 17, text, color, edge, 10)
    for x, text in rejected:
        _diagram_box(ax, x, 31, 14, 14, text, "#FFEDD5", ORANGE, 9)
        _diagram_arrow(ax, (x, 39), (x, 57), color=ORANGE, dashed=True)
    for (left, _), (right, _) in zip(adopted, adopted[1:]):
        _diagram_arrow(ax, (left + 7.5, 66), (right - 7.5, 66), color=BLUE)
    ax.text(
        1,
        87,
        "Autoresearch decision path",
        ha="left",
        va="top",
        color=INK,
        fontsize=17,
        fontweight="bold",
    )
    ax.text(1, 52, "Adopted", ha="left", color=GREEN, fontsize=11, fontweight="bold")
    ax.text(
        1,
        19,
        "Rejected / control",
        ha="left",
        color=ORANGE,
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        50,
        5,
        "Stage A: mask granularity   ·   Stage B: latent stability + predictor warm start",
        ha="center",
        color=MUTED,
        fontsize=10,
        style="italic",
    )
    _save(fig, "exploration_decision_path.svg")


def plot_mask_comparison(frame_test, token_test):
    labels = ["Validation", "Test"]
    frame_joint = [0.306420, frame_test["latent_joint_relative"]]
    token_joint = [0.289151, token_test["latent_joint_relative"]]
    frame_future = [0.337029, frame_test["latent_relative"]]
    token_future = [0.318035, token_test["latent_relative"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=BG)
    for ax, frame_values, token_values, title in (
        (axes[0], frame_joint, token_joint, "Joint context + future loss"),
        (axes[1], frame_future, token_future, "Future latent loss"),
    ):
        x = np.arange(2)
        width = 0.34
        b1 = ax.bar(
            x - width / 2,
            frame_values,
            width,
            label="Whole future frame",
            color="#94A3B8",
            edgecolor="white",
        )
        b2 = ax.bar(
            x + width / 2,
            token_values,
            width,
            label="Partial spatial tokens",
            color=CYAN,
            edgecolor="white",
        )
        ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=9, color=MUTED)
        ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=9, color=INK)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 0.42)
        ax.set_ylabel("Relative loss (lower is better)")
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
        _style(ax)
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle(
        "Stage A: spatial-token masking wins both objectives",
        x=0.06,
        ha="left",
        color=INK,
        fontsize=16,
        fontweight="bold",
    )
    _save(fig, "stage_a_mask_comparison.svg")


def plot_mask_schematic():
    """Show the fine encoder grid separately from the coarse prediction grid."""
    rows, cols = 14, 21
    fig, ax = plt.subplots(figsize=(15, 9.5), facecolor=BG)
    ax.set_xlim(0, 29)
    ax.set_ylim(0, 25)
    ax.axis("off")

    def draw_frame(x, y, width, height, visible, label):
        ax.text(
            x + width / 2,
            y + height + 0.3,
            label,
            ha="center",
            va="bottom",
            color=MUTED,
            fontsize=10,
        )
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.12, y - 0.12),
                width + 0.24,
                height + 0.24,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor="white",
                edgecolor="#CBD5E1",
                linewidth=1.0,
            )
        )
        cell_w = width / cols
        cell_h = height / rows
        for row in range(rows):
            for col in range(cols):
                show = bool(visible[row, col])
                ax.add_patch(
                    Rectangle(
                        (x + col * cell_w, y + (rows - 1 - row) * cell_h),
                        cell_w * 0.92,
                        cell_h * 0.88,
                        facecolor="#67E8F9" if show else "#D8E0EA",
                        edgecolor="none",
                    )
                )

    ax.text(
        0.8,
        24.1,
        "Fine encoder tokenization",
        ha="left",
        color=INK,
        fontsize=14,
        fontweight="bold",
    )
    full_visible = np.ones((rows, cols), dtype=bool)
    draw_frame(1.0, 17.1, 7.0, 5.7, full_visible, "one 336×224 GelSight frame")
    ax.annotate(
        "",
        xy=(13.4, 18.3),
        xytext=(8.4, 19.8),
        arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": 1.8},
    )
    ax.text(
        13.7,
        20.0,
        "patch = 16×16, tubelet = 1\n21×14 = 294 ViT tokens / hand\n588 tokens / frame for two hands",
        ha="left",
        va="center",
        color=INK,
        fontsize=11,
        linespacing=1.5,
    )
    ax.text(
        13.7,
        17.3,
        "No pixel decoder: both mask ablations predict the same\n"
        "30-frame coarse latent target. The 2×2 grid is post-encoding.",
        ha="left",
        va="center",
        color=MUTED,
        fontsize=10,
        linespacing=1.35,
        style="italic",
    )

    future_x = [0.8 + index * 4.45 for index in range(5)]
    ax.text(
        0.8,
        15.0,
        "future_frames control: the online encoder sees no future fine token",
        ha="left",
        color=INK,
        fontsize=13,
        fontweight="bold",
    )
    for index, x in enumerate(future_x):
        if index == 0:
            visible = full_visible
            label = "context · unmasked"
        else:
            visible = np.zeros((rows, cols), dtype=bool)
            label = f"future t+{index}"
        draw_frame(x, 9.1, 3.25, 4.2, visible, label)

    ax.text(
        0.8,
        7.1,
        "spatial_tokens selected: each future frame independently masks 75% of its fine tokens",
        ha="left",
        color=INK,
        fontsize=13,
        fontweight="bold",
    )
    for index, x in enumerate(future_x):
        if index == 0:
            visible = full_visible
            label = "context · unmasked"
        else:
            rng = np.random.default_rng(7919 + index)
            visible = rng.random((rows, cols)) >= 0.75
            visible[0, 0] = True
            label = f"future t+{index}"
        draw_frame(x, 1.2, 3.25, 4.2, visible, label)

    ax.add_patch(
        Rectangle(
            (23.8, 4.1),
            0.35,
            0.35,
            facecolor="#67E8F9",
            edgecolor=CYAN,
        )
    )
    ax.text(24.35, 4.27, "visible fine token", va="center", color=MUTED, fontsize=9)
    ax.add_patch(
        Rectangle(
            (23.8, 3.3),
            0.35,
            0.35,
            facecolor="#D8E0EA",
            edgecolor="#94A3B8",
        )
    )
    ax.text(24.35, 3.47, "mask token", va="center", color=MUTED, fontsize=9)

    fig.suptitle(
        "Stage A input-mask ablation: same 30-frame future latent target, different visibility",
        x=0.04,
        ha="left",
        color=INK,
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.04,
        0.015,
        "One shared hand encoder is shown. All 4 context frames stay unmasked in both branches. "
        "No raw future frame is reconstructed; the EMA encoder supplies all future latent targets.",
        color=MUTED,
        fontsize=9,
    )
    _save(fig, "mask_granularity_schematic.svg")


def plot_stage_a_convergence(frame_metrics, token_metrics):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
    series = (
        ("Whole future frame", frame_metrics, "#94A3B8"),
        ("Partial spatial tokens", token_metrics, CYAN),
    )
    for label, payload, color in series:
        records = payload["records"]
        steps = sorted(int(step) for step in records)
        joint = [
            float(records[str(step)]["validation"]["latent_joint_relative"])
            for step in steps
        ]
        future = [
            float(records[str(step)]["validation"]["latent_relative"])
            for step in steps
        ]
        axes[0].plot(steps, joint, label=label, color=color, linewidth=2.2)
        axes[0].scatter(
            [int(payload["best_step"])],
            [min(joint)],
            s=70,
            color=color,
            edgecolor="white",
            linewidth=1.2,
            zorder=5,
        )
        axes[1].plot(steps, future, label=label, color=color, linewidth=2.2)
    axes[0].set_title(
        "Validation joint relative", loc="left", fontsize=13, fontweight="bold"
    )
    axes[1].set_title(
        "Validation future relative", loc="left", fontsize=13, fontweight="bold"
    )
    for ax in axes:
        ax.set_xlabel("Optimization step")
        ax.set_ylabel("Relative loss (lower is better)")
        ax.set_ylim(0.25, 0.82)
        _style(ax)
    axes[0].legend(frameon=False, loc="upper right", fontsize=9)
    fig.suptitle(
        "Stage A convergence under deterministic validation masks",
        x=0.06,
        ha="left",
        color=INK,
        fontsize=16,
        fontweight="bold",
    )
    _save(fig, "stage_a_convergence.svg")


def plot_stage_b_strategy_comparison(random_metrics, warm_metrics, frozen_metrics):
    runs = (
        ("Random predictor + encoder tuning", random_metrics, ORANGE),
        ("Stage-A warm start + encoder tuning", warm_metrics, PURPLE),
        ("Stage-A warm start + frozen latent", frozen_metrics, GREEN),
    )
    fig, ax = plt.subplots(figsize=(12.5, 5.2), facecolor=BG)
    for label, payload, color in runs:
        records = payload["records"]
        steps = sorted(int(step) for step in records)
        values = [
            float(records[str(step)]["validation"]["tactile_relative"])
            for step in steps
        ]
        ax.plot(
            steps,
            values,
            color=color,
            linewidth=2.3,
            marker="o",
            markersize=4,
            label=label,
        )
        if steps[-1] < 1000:
            ax.annotate(
                f"aborted @ {steps[-1]}",
                (steps[-1], values[-1]),
                xytext=(-8, 18),
                textcoords="offset points",
                ha="right",
                color=color,
                fontsize=9,
                fontweight="bold",
            )
    ax.set_xlim(0, 2550)
    ax.set_ylim(0.18, 0.86)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Validation tactile relative (lower is better)")
    ax.set_title(
        "Stage B strategy comparison: latent stability matters most",
        loc="left",
        color=INK,
        fontsize=16,
        fontweight="bold",
    )
    ax.legend(frameon=False, loc="upper center", fontsize=9)
    _style(ax)
    _save(fig, "stage_b_strategy_comparison.svg")


def plot_stage_b_convergence(metrics):
    records = metrics["records"]
    steps = sorted(int(step) for step in records)
    tactile = [
        float(records[str(step)]["validation"]["tactile_relative"])
        for step in steps
    ]
    score = [float(records[str(step)]["score"]) for step in steps]
    best = int(metrics["best_step"])
    best_value = tactile[steps.index(best)]

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
    ax.plot(steps, tactile, color=BLUE, linewidth=2.3, label="Tactile relative")
    ax.plot(steps, score, color=PURPLE, linewidth=1.6, alpha=0.75, label="Selection score")
    ax.axvline(best, color=GREEN, linestyle="--", linewidth=1.4, label="Best checkpoint")
    ax.scatter(
        [best],
        [best_value],
        s=95,
        color=GREEN,
        edgecolor="white",
        linewidth=1.5,
        zorder=5,
    )
    ax.annotate(
        f"best {best_value:.3f} @ {best}",
        (best, best_value),
        xytext=(18, 20),
        textcoords="offset points",
        color=GREEN,
        fontsize=10,
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": GREEN, "lw": 1.1},
    )
    ax.annotate(
        "early stop @ 2500",
        (2500, tactile[steps.index(2500)]),
        xytext=(-70, 32),
        textcoords="offset points",
        color=ORANGE,
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.1},
    )
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Validation metric")
    ax.set_title(
        "Stage B: fixed Stage-A latent space",
        loc="left",
        color=INK,
        fontsize=16,
        fontweight="bold",
    )
    ax.legend(frameon=False, loc="upper right")
    _style(ax)
    _save(fig, "stage_b_convergence.svg")


def plot_test_metrics(test):
    labels = ["Tactile", "Event", "Planning", "Visual", "Contact\nprobe", "Slip\nprobe"]
    values = [
        test["tactile_relative"],
        test["event_relative"],
        test["planning_relative"],
        test["visual_relative"],
        test["probe_contact_auroc"],
        test["probe_slip_auroc"],
    ]
    colors = [GREEN, CYAN, BLUE, ORANGE, PURPLE, "#DB2777"]
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.66)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=10, color=INK)
    ax.set_ylim(0, 0.86)
    ax.set_ylabel("Score")
    ax.set_title(
        "Final Stage B held-out test",
        loc="left",
        color=INK,
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        0,
        0.93,
        "Lower is better for relative-loss metrics; higher is better for AUROC.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9,
    )
    _style(ax)
    _save(fig, "stage_b_test_metrics.svg")


def plot_horizon():
    offsets = list(range(1, 31))
    relative = [
        0.1468, 0.1687, 0.1787, 0.1780, 0.1748, 0.1744, 0.1866, 0.1977,
        0.2024, 0.1988, 0.1931, 0.1912, 0.2022, 0.2130, 0.2177, 0.2166,
        0.2131, 0.2133, 0.2249, 0.2340, 0.2366, 0.2342, 0.2317, 0.2287,
        0.2382, 0.2451, 0.2500, 0.2483, 0.2465, 0.2400,
    ]
    cosine = [
        0.9774, 0.9723, 0.9694, 0.9699, 0.9708, 0.9708, 0.9675, 0.9644,
        0.9625, 0.9632, 0.9649, 0.9653, 0.9623, 0.9588, 0.9570, 0.9566,
        0.9579, 0.9578, 0.9542, 0.9511, 0.9503, 0.9508, 0.9522, 0.9527,
        0.9500, 0.9478, 0.9459, 0.9462, 0.9469, 0.9491,
    ]
    fig, ax1 = plt.subplots(figsize=(13, 5.2), facecolor=BG)
    ax2 = ax1.twinx()
    ax1.plot(offsets, relative, color=BLUE, linewidth=2.4, marker="o", markersize=4)
    ax2.plot(offsets, cosine, color=ORANGE, linewidth=2.2, linestyle="--")
    ax1.fill_between(offsets, relative, 0, color=BLUE, alpha=0.08)
    ax1.set_xlabel("Future horizon (frames)")
    ax1.set_ylabel("Relative loss (lower is better)", color=BLUE)
    ax2.set_ylabel("Delta cosine (higher is better)", color=ORANGE)
    ax1.set_ylim(0.10, 0.28)
    ax2.set_ylim(0.92, 1.0)
    ax1.set_title(
        "Future tactile latent stability by horizon",
        loc="left",
        color=INK,
        fontsize=16,
        fontweight="bold",
    )
    _style(ax1)
    ax2.spines[["top", "left"]].set_visible(False)
    ax2.spines["right"].set_color("#F6C7A8")
    ax2.tick_params(colors=ORANGE, labelsize=10)
    ax1.annotate(
        "t+1  0.147 / 0.977",
        (1, relative[0]),
        xytext=(8, 18),
        textcoords="offset points",
        fontsize=9,
        color=BLUE,
    )
    ax1.annotate(
        "t+30  0.240 / 0.949",
        (30, relative[-1]),
        xytext=(-88, -28),
        textcoords="offset points",
        fontsize=9,
        color=BLUE,
    )
    _save(fig, "horizon_stability.svg")


def plot_query_tokens(condition):
    probes = condition["independent_ridge_probes"]
    labels = ["History\ntactile", "Future\ntactile", "History\nvisual", "Future\nvisual", "History\nstate"]
    keys = ["history_tactile", "future_tactile", "history_visual", "future_visual", "history_state"]
    r2 = [probes[key]["test_r2"] for key in keys]
    cosine = [probes[key]["test_cosine"] for key in keys]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
    bars = axes[0].bar(labels, r2, color=[GREEN, CYAN, BLUE, PURPLE, ORANGE], edgecolor="white")
    axes[0].bar_label(bars, fmt="%.3f", padding=4, fontsize=9, color=INK)
    axes[0].set_ylim(0.45, 1.01)
    axes[0].set_ylabel("Test R²")
    axes[0].set_title("Independent linear decodability", loc="left", fontsize=12, fontweight="bold")
    bars = axes[1].bar(labels, cosine, color=[GREEN, CYAN, BLUE, PURPLE, ORANGE], edgecolor="white")
    axes[1].bar_label(bars, fmt="%.3f", padding=4, fontsize=9, color=INK)
    axes[1].set_ylim(0.992, 1.0004)
    axes[1].set_ylabel("Test cosine")
    axes[1].set_title("Normalized direction agreement", loc="left", fontsize=12, fontweight="bold")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
        _style(ax)
    fig.suptitle(
        "Four condition tokens are information-rich",
        x=0.06,
        ha="left",
        color=INK,
        fontsize=16,
        fontweight="bold",
    )
    _save(fig, "query_token_information.svg")


def plot_query_redundancy():
    labels = ["Slot 0", "Slot 1", "Slot 2", "Slot 3", "All 4", "Drop 0", "Drop 1", "Drop 2", "Drop 3"]
    values = [
        0.976986, 0.977525, 0.976823, 0.977329, 0.978741,
        0.978683, 0.978418, 0.978702, 0.978418,
    ]
    colors = [CYAN] * 4 + [GREEN] + ["#94A3B8"] * 4
    fig, ax = plt.subplots(figsize=(12, 4.8), facecolor=BG)
    bars = ax.bar(labels, values, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8, color=INK)
    ax.set_ylim(0.972, 0.981)
    ax.set_ylabel("Future tactile test R²")
    ax.set_title(
        "Each query already carries the target; leave-one-out is nearly flat",
        loc="left",
        color=INK,
        fontsize=15,
        fontweight="bold",
    )
    _style(ax)
    _save(fig, "query_token_redundancy.svg")


def plot_query_geometry(condition):
    metrics = condition["condition_tokens"]
    cosine = np.asarray(metrics["token_centroid_pairwise_cosine"], dtype=float)
    norms = np.asarray(metrics["token_norm_by_slot"], dtype=float)
    effective_rank = float(metrics["sample_centered_effective_rank"])
    mean_cosine = float(metrics["test_pairwise_cosine_mean"])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 5),
        facecolor=BG,
        gridspec_kw={"width_ratios": [1.25, 0.75]},
    )
    image = axes[0].imshow(
        cosine,
        cmap="RdBu_r",
        vmin=-0.12,
        vmax=0.12,
        interpolation="nearest",
    )
    for row in range(cosine.shape[0]):
        for col in range(cosine.shape[1]):
            value = cosine[row, col]
            color = "white" if abs(value) > 0.07 else INK
            axes[0].text(
                col,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
                fontweight="bold" if row != col else "normal",
            )
    axes[0].set_xticks(range(4), [f"Q{i}" for i in range(4)])
    axes[0].set_yticks(range(4), [f"Q{i}" for i in range(4)])
    axes[0].set_title(
        "Token-centroid cosine similarity",
        loc="left",
        color=INK,
        fontsize=12,
        fontweight="bold",
    )
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(colors=MUTED, labelsize=8)
    colorbar.outline.set_edgecolor("#CBD5E1")
    axes[0].text(
        0.02,
        1.02,
        f"mean pairwise {mean_cosine:.3f} · effective rank {effective_rank:.1f}",
        transform=axes[0].transAxes,
        color=MUTED,
        fontsize=9,
    )

    bars = axes[1].bar(
        [f"Q{i}" for i in range(4)],
        norms,
        color=[GREEN, CYAN, BLUE, PURPLE],
        edgecolor="white",
        width=0.66,
    )
    axes[1].bar_label(bars, fmt="%.2f", padding=3, fontsize=9, color=INK)
    axes[1].set_ylim(20.5, 21.5)
    axes[1].set_ylabel("Mean token norm")
    axes[1].set_title(
        "No norm collapse",
        loc="left",
        color=INK,
        fontsize=12,
        fontweight="bold",
    )
    _style(axes[1])
    axes[1].grid(axis="x", visible=False)
    fig.suptitle(
        "Four query tokens: diverse directions, balanced magnitude, high-rank set",
        x=0.06,
        ha="left",
        color=INK,
        fontsize=16,
        fontweight="bold",
    )
    _save(fig, "query_token_geometry.svg")


def _strategy_summary(name: str, metrics):
    records = metrics["records"]
    steps = sorted(int(step) for step in records)
    values = [
        float(records[str(step)]["validation"]["tactile_relative"])
        for step in steps
    ]
    best_index = int(np.argmin(values))
    return {
        "name": name,
        "best_observed_step": steps[best_index],
        "best_observed_validation_tactile_relative": values[best_index],
        "last_validation_step": steps[-1],
        "last_validation_tactile_relative": values[-1],
    }


def main():
    frame_a = _load(RUN / "comparisons/frame_mask/metrics.json")
    token_a = _load(RUN / "stage_a/metrics.json")
    frame_test = _load(RUN / "evaluation/FORMAL_TEST_FRAME.json")["metrics"]
    token_test = _load(RUN / "evaluation/FORMAL_TEST_TOKEN.json")["metrics"]
    stage_b_metrics = _load(RUN / "stage_b/metrics.json")
    stage_b_random = _load(RUN / "comparisons/stage_b_random/metrics.json")
    stage_b_warm = _load(RUN / "comparisons/stage_b_warm/metrics.json")
    stage_b_test = _load(RUN / "stage_b/test_metrics.json")["test_metrics"]
    condition = _load(RUN / "evaluation/FINAL_CONDITION_TOKENS.json")

    summary = {
        "stage_a": {
            "future_frame_best_step": frame_a["best_step"],
            "future_frame_best_score": frame_a["best_score"],
            "spatial_token_best_step": token_a["best_step"],
            "spatial_token_best_score": token_a["best_score"],
            "future_frame_test": frame_test,
            "spatial_token_test": token_test,
        },
        "stage_b": {
            "best_step": stage_b_metrics["best_step"],
            "best_score": stage_b_metrics["best_score"],
            "stopped_at": max(int(step) for step in stage_b_metrics["records"]),
            "test": stage_b_test,
            "strategy_ablation": [
                _strategy_summary(
                    "random_predictor_encoder_tuning", stage_b_random
                ),
                _strategy_summary(
                    "stage_a_warm_start_encoder_tuning", stage_b_warm
                ),
                _strategy_summary(
                    "stage_a_warm_start_frozen_latent", stage_b_metrics
                ),
            ],
        },
        "condition_tokens": {
            "metrics": condition["condition_tokens"],
            "independent_ridge_probes": condition["independent_ridge_probes"],
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    plot_mask_comparison(frame_test, token_test)
    plot_mask_schematic()
    plot_stage_a_convergence(frame_a, token_a)
    plot_stage_b_strategy_comparison(
        stage_b_random, stage_b_warm, stage_b_metrics
    )
    plot_stage_b_convergence(stage_b_metrics)
    plot_test_metrics(stage_b_test)
    plot_horizon()
    plot_query_tokens(condition)
    plot_query_redundancy()
    plot_query_geometry(condition)
    plot_architecture_overview()
    plot_decision_path()
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
