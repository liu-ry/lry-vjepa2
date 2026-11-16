"""Render the deployed Stage-B inference architecture as a PNG."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "stage_b_deployment_architecture.png"
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")

if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))
    font_name = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
    plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False


COLORS = {
    "bg": "#F8FAFC",
    "ink": "#0F172A",
    "muted": "#475569",
    "grid": "#CBD5E1",
    "input": ("#E0F2FE", "#0284C7"),
    "tactile": ("#DCFCE7", "#16A34A"),
    "visual": ("#DBEAFE", "#2563EB"),
    "predictor": ("#FEF3C7", "#D97706"),
    "output": ("#FCE7F3", "#DB2777"),
    "note": ("#F1F5F9", "#64748B"),
}


def box(
    ax,
    x,
    y,
    width,
    height,
    title,
    body="",
    *,
    fill,
    edge,
    title_size=15,
    body_size=11,
    dashed=False,
    radius=0.12,
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.045,rounding_size={radius}",
        facecolor=fill,
        edgecolor=edge,
        linewidth=1.8,
        linestyle="--" if dashed else "-",
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * (0.66 if body else 0.5),
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=3,
    )
    if body:
        ax.text(
            x + width / 2,
            y + height * 0.30,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color=COLORS["muted"],
            linespacing=1.35,
            zorder=3,
        )
    return (x, y, width, height)


def arrow(ax, start, end, *, color="#64748B", width=2.0, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=width,
            color=color,
            linestyle="--" if dashed else "-",
            shrinkA=4,
            shrinkB=4,
            connectionstyle="arc3,rad=0.0",
            zorder=1,
        )
    )


def right(rect):
    return rect[0] + rect[2], rect[1] + rect[3] / 2


def left(rect):
    return rect[0], rect[1] + rect[3] / 2


def top(rect):
    return rect[0] + rect[2] / 2, rect[1] + rect[3]


def bottom(rect):
    return rect[0] + rect[2] / 2, rect[1]


def main():
    fig, ax = plt.subplots(figsize=(16, 10.2), dpi=180)
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(
        0.45,
        9.58,
        "Stage B 真实部署推理架构",
        fontsize=25,
        fontweight="bold",
        color=COLORS["ink"],
        ha="left",
        va="center",
    )
    ax.text(
        0.47,
        9.18,
        "输入仅使用 4 帧历史 · 无 action 输入 / 输出 · 执行前向参数 137.74M",
        fontsize=13,
        color=COLORS["muted"],
        ha="left",
        va="center",
    )

    # Inputs
    ax.text(
        0.55,
        8.63,
        "部署输入",
        fontsize=13,
        fontweight="bold",
        color=COLORS["input"][1],
    )
    tactile_in = box(
        ax,
        0.45,
        6.55,
        2.25,
        1.35,
        "触觉输入",
        "4 chains × 4 frames\n3 × 336 × 224",
        fill=COLORS["input"][0],
        edge=COLORS["input"][1],
    )
    visual_in = box(
        ax,
        0.45,
        4.78,
        2.25,
        1.35,
        "视觉输入",
        "2 arms × 4 frames\n3 × 384 × 384",
        fill=COLORS["input"][0],
        edge=COLORS["input"][1],
    )
    state_in = box(
        ax,
        0.45,
        3.18,
        2.25,
        1.05,
        "机器人状态",
        "2 arms × 4 frames × 8D\n归一化：0 params",
        fill=COLORS["note"][0],
        edge=COLORS["note"][1],
        dashed=True,
    )

    # Encoders
    ax.text(
        3.08,
        8.63,
        "编码器（共享权重）",
        fontsize=13,
        fontweight="bold",
        color=COLORS["ink"],
    )
    tactile_encoder = box(
        ax,
        3.0,
        6.28,
        3.25,
        1.78,
        "在线触觉编码器  ·  14.914M",
        "Frame ViT 11.351M  +  Causal Temporal 3.563M\n"
        "4-frame context → 7×7 pool  ·  [4×4×49×384]",
        fill=COLORS["tactile"][0],
        edge=COLORS["tactile"][1],
        title_size=14,
    )
    visual_encoder = box(
        ax,
        3.0,
        4.48,
        3.25,
        1.55,
        "V-JEPA2.1 ViT-B  ·  86.242M active",
        "12 layers · 768D · video branch only\n"
        "4 frames → 2 temporal steps · [2×2×576×768]",
        fill=COLORS["visual"][0],
        edge=COLORS["visual"][1],
        title_size=14,
    )
    ax.text(
        3.18,
        3.86,
        "视觉 image patch 分支 0.591M 当前不执行",
        fontsize=9.5,
        color=COLORS["visual"][1],
        ha="left",
        va="center",
        style="italic",
    )

    # Predictor panel
    panel = FancyBboxPatch(
        (6.65, 1.55),
        8.85,
        6.72,
        boxstyle="round,pad=0.055,rounding_size=0.16",
        facecolor="#FFFDF5",
        edgecolor=COLORS["predictor"][1],
        linewidth=2.2,
        zorder=0,
    )
    ax.add_patch(panel)
    ax.text(
        6.92,
        7.96,
        "ChainWorldPredictor  ·  36.586M",
        fontsize=17,
        fontweight="bold",
        color=COLORS["predictor"][1],
        ha="left",
        va="center",
    )

    projector = box(
        ax,
        6.98,
        6.50,
        3.15,
        1.05,
        "输入投影 + Token Embeddings",
        "Tactile / Visual / State MLP\n"
        "tokens · chain · arm · time · spatial  ·  0.929M",
        fill=COLORS["predictor"][0],
        edge=COLORS["predictor"][1],
        title_size=12.5,
        body_size=9.8,
    )
    context = box(
        ax,
        6.98,
        5.02,
        3.15,
        1.08,
        "Context + Cross-Modal Fusion",
        "3L Tactile Transformer\n"
        "Visual Fusion + State Fusion  ·  6.508M",
        fill=COLORS["predictor"][0],
        edge=COLORS["predictor"][1],
        title_size=12.5,
        body_size=9.8,
    )
    future = box(
        ax,
        6.98,
        3.62,
        3.15,
        1.02,
        "Future Query Decoding",
        "Cross-Attention + Dense Heads\n"
        "tactile 30 frames / visual 15 frames  ·  1.333M",
        fill=COLORS["predictor"][0],
        edge=COLORS["predictor"][1],
        title_size=12.5,
        body_size=9.8,
    )
    queries = box(
        ax,
        6.98,
        2.22,
        3.15,
        1.02,
        "4 × Condition Queries",
        "Condition Attention + 2L Transformer\n"
        "4 query tokens  ·  2.960M",
        fill=COLORS["predictor"][0],
        edge=COLORS["predictor"][1],
        title_size=12.5,
        body_size=9.8,
    )

    heads = box(
        ax,
        10.62,
        2.20,
        4.47,
        4.42,
        "",
        "",
        fill="#FFF7DB",
        edge=COLORS["predictor"][1],
        title_size=15,
    )
    ax.text(
        12.855,
        6.31,
        "输出解码与 Summary Heads",
        fontsize=15,
        fontweight="bold",
        color=COLORS["ink"],
        ha="center",
        va="center",
        zorder=3,
    )
    head_lines = [
        ("History tactile summary", "2.514M"),
        ("Future tactile summary", "2.514M"),
        ("Future trajectory segments", "18.335M"),
        ("Future visual summary", "1.331M"),
        ("History state summary", "0.161M"),
    ]
    for index, (label, value) in enumerate(head_lines):
        y = 5.94 - index * 0.67
        ax.text(10.92, y, label, fontsize=10.8, color=COLORS["ink"], ha="left")
        ax.text(14.78, y, value, fontsize=10.8, color=COLORS["predictor"][1], ha="right", fontweight="bold")
        ax.plot([10.90, 14.79], [y - 0.20, y - 0.20], color="#F3D58C", linewidth=0.8)
    ax.text(
        12.855,
        2.52,
        "Heads subtotal  ·  24.856M",
        fontsize=11.8,
        fontweight="bold",
        color=COLORS["predictor"][1],
        ha="center",
        va="center",
    )

    # Outputs
    out_tactile = box(
        ax,
        10.62,
        1.02,
        2.18,
        0.88,
        "future_tactile",
        "[B, 4, 30, 49, 384]",
        fill=COLORS["output"][0],
        edge=COLORS["output"][1],
        title_size=11.5,
        body_size=9.5,
    )
    out_visual = box(
        ax,
        12.98,
        1.02,
        2.11,
        0.88,
        "future_visual",
        "[B, 2, 15, 768]",
        fill=COLORS["output"][0],
        edge=COLORS["output"][1],
        title_size=11.5,
        body_size=9.5,
    )
    out_query = box(
        ax,
        7.00,
        1.02,
        3.13,
        0.88,
        "condition_tokens",
        "[B, 4, 384]  ·  downstream interface",
        fill=COLORS["output"][0],
        edge=COLORS["output"][1],
        title_size=11.5,
        body_size=9.1,
    )

    # Main arrows
    arrow(ax, right(tactile_in), left(tactile_encoder), color=COLORS["tactile"][1], width=2.6)
    arrow(ax, right(visual_in), left(visual_encoder), color=COLORS["visual"][1], width=2.6)
    arrow(ax, right(tactile_encoder), (6.98, 7.02), color=COLORS["tactile"][1], width=2.6)
    arrow(ax, right(visual_encoder), (6.98, 6.67), color=COLORS["visual"][1], width=2.6)
    arrow(ax, right(state_in), (6.98, 6.92), color=COLORS["note"][1], width=2.0, dashed=True)
    arrow(ax, bottom(projector), top(context), color=COLORS["predictor"][1], width=2.1)
    arrow(ax, bottom(context), top(future), color=COLORS["predictor"][1], width=2.1)
    arrow(ax, bottom(future), top(queries), color=COLORS["predictor"][1], width=2.1)
    arrow(ax, right(future), (10.62, 4.13), color=COLORS["predictor"][1], width=2.1)
    arrow(ax, right(queries), (10.62, 2.78), color=COLORS["predictor"][1], width=2.1)
    arrow(ax, (12.83, 2.20), top(out_tactile), color=COLORS["output"][1], width=2.0)
    arrow(ax, (14.04, 2.20), top(out_visual), color=COLORS["output"][1], width=2.0)
    arrow(ax, bottom(queries), top(out_query), color=COLORS["output"][1], width=2.0)

    # Total and deployment note
    total_box = FancyBboxPatch(
        (0.45, 0.13),
        7.45,
        0.63,
        boxstyle="round,pad=0.035,rounding_size=0.10",
        facecolor="#ECFDF5",
        edgecolor="#059669",
        linewidth=1.6,
        zorder=2,
    )
    ax.add_patch(total_box)
    ax.text(
        4.175,
        0.445,
        "部署执行参数 = 14.914M + 86.242M + 36.586M  =  137.742M",
        fontsize=12.3,
        fontweight="bold",
        color="#065F46",
        ha="center",
        va="center",
    )
    note_box = FancyBboxPatch(
        (8.25, 0.13),
        7.28,
        0.63,
        boxstyle="round,pad=0.035,rounding_size=0.10",
        facecolor=COLORS["note"][0],
        edgecolor=COLORS["note"][1],
        linewidth=1.3,
        zorder=2,
    )
    ax.add_patch(note_box)
    ax.text(
        11.89,
        0.445,
        "当前 Extractor 额外加载但未执行：target 14.914M + image branch 0.591M + probes 0.0046M",
        fontsize=9.5,
        color=COLORS["muted"],
        ha="center",
        va="center",
    )

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT,
        dpi=180,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.12,
    )
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
