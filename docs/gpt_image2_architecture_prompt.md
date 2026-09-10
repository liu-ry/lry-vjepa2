# GPT-Image-2 prompt — VLM-centered architecture (revised)

```text
Use case: scientific-educational
Asset type: publication-quality robotics / multimodal learning architecture figure
Input image: use the attached reference image only as a visual-style and layout reference: clean paper figure, pastel modules, token tiles, compact arrows. Do not copy its labels or invent unspecified modules.

Create a restrained, publication-quality 16:9 landscape architecture figure in the flat, compact scientific style of modern robotics papers. Use a white background, mostly rectangular panels with thin gray/black outlines, flat low-saturation fills, restrained color coding, small paper-style typography, and short orthogonal arrows. Avoid a glossy generative-AI look: no gradients, 3D globes, drop shadows, emoji, flames, dice, cartoon sensors, giant decorative title, excessive rounded cards, or photorealistic imagery. Use simple abstract frame thumbnails and token rectangles only. The PRIMARY SUBJECT is a large central-left blue panel labeled "Vision-Language Model (VLM)" and a large central-right orange panel labeled "Action Expert". The newly added V-JEPA 2.1 world model is a smaller AUXILIARY MODULE above them, visually subordinate in size. The bottom residual controller is compact. Do not use the string "π0.5" anywhere in the image. Render English labels exactly as written below. No watermark, logo, people, decorative arrows, missing arrows, or incorrect arrowheads.

TITLE at the top center:
"A Vision–Tactile World-Model-Based VTLA Architecture for Residual Action Correction"

============================================================
PRIMARY PATH: VLM + ACTION EXPERT (largest central horizontal path)
============================================================

LEFT INPUT PANEL, labeled "CURRENT OBSERVATION": show exactly three current single-step inputs, not histories:
"Current RGB frame"
"Current tactile frame"
"Current proprioceptive state"
Draw EXACTLY three arrows from these current inputs directly into a simple large blue box labeled only "Vision-Language Model (VLM)". Do not put internal sub-blocks, token rows, encoder names, or embeddings inside the VLM box. Do NOT draw four-frame history stacks into the VLM. Do NOT insert the V-JEPA, tactile, or proprioceptive encoders inside the VLM; those encoders belong to the separate world-model history branch described below.

Inside or immediately after the VLM show a small blue-gray block labeled "KV-cache context". Draw one arrow from "Vision-Language Model (VLM)" → "KV-cache context" → "Action Expert". The KV-cache carries VLM context to the Action Expert. Do not draw a direct history-token input to the Action Expert.

The large orange panel is labeled "Action Expert" and contains "Original action chunk a⁰[1:30]" and "Generated once at initialization". It receives the VLM KV-cache context and the future latent plan. Draw one downward arrow from this original action chunk to the execution timeline. Do not show the VLM or Action Expert called again during execution.

============================================================
AUXILIARY WORLD MODEL ABOVE THE VLM / ACTION EXPERT
============================================================

Place a compact green panel ABOVE the VLM and Action Expert, labeled:
"Auxiliary History-only V-JEPA 2.1 Latent World Model"
Add the badge:
"Auxiliary guidance — not the action generator"

Above or beside the current-observation panel, show a separate dashed history branch labeled "WORLD-MODEL HISTORY ONLY" with "RGB history (4 frames)", "Tactile history (4 frames)", and "Proprioceptive state history (4 steps)". On this separate branch, explicitly draw three clearly visible encoder modules before the Transformer, each as its own rectangular block:
"Frozen V-JEPA 2.1 Visual Encoder"
"Trainable Tactile Encoder"
"Trainable Proprioceptive State Encoder"
The outputs are labeled "visual latent context", "tactile latent context", and "state tokens". Draw EXACTLY three arrows from these three encoder outputs into the auxiliary world model. These history arrows must NOT enter the VLM. Inside show only "History-only Multimodal Transformer" and the badge "No action input". Do NOT draw an action-chunk arrow into this world model.

Inside the world-model panel add a plain dashed TRAINING ONLY box (text only, no dice, no decorative icon):
"TRAINING ONLY"
"Random seed"
"sample 6 sorted future offsets per batch"
"example: [3, 6, 10, 20, 25, 30]"
This box points to exactly six query tokens: "τ1 τ2 τ3 τ4 τ5 τ6". Add "same six time points for visual and tactile prediction".

Show two output rows:
"Predicted visual latent plan: ẑv(τ1)...ẑv(τ6)"
"Predicted tactile latent plan: ẑh(τ1)...ẑh(τ6)"
Draw EXACTLY ONE and only one green arrow from the combined output "Future latent plan" directly to ONE clearly marked conditioning socket inside the Action Expert. Label it "latent guidance". Do not draw any latent-guidance arrow into the VLM or KV-cache. Do not draw any second green arrow to the original action chunk or execution timeline. The Action Expert receives two inputs: VLM context through the KV-cache and the future latent plan.

Add a dashed INFERENCE ONLY note near the output:
"INFERENCE: choose offsets explicitly"
"deployment example: [5, 10, 15, 20, 25, 30]"
The random-seed box is training-only and must not look active at inference.

============================================================
CLOSED-LOOP EXECUTION AND RESIDUAL CORRECTION (bottom full-width panel)
============================================================

Create a compact full-width red/purple panel labeled "ONLINE EXECUTION AND RESIDUAL ACTION CORRECTION". Do NOT draw six repeated checkpoint cards, three separate bottom sub-panels, a separate "execute-correct loop" box, or a second copy of the residual controller. Use one slim 30-slot action timeline grouped into six groups of five, and ONE compact representative correction schematic directly below or beside the timeline.
Label the timeline only with compact group labels, not six expanded mini-workflows:
"1–5"  "6–10"  "11–15"  "16–20"  "21–25"  "26–30"
Add one centered caption above it:
"Execute 5 actions → observe → compare → correct remaining actions (repeat at τ1…τ5)"
At the far right add: "τ6: final observation only"

Show small checkpoint markers τ1, τ2, τ3, τ4, τ5, τ6 along the timeline. Do not duplicate the encoder/comparison diagram for each marker.

Show ONE representative correction schematic labeled "Representative checkpoint: t=τi (every 5 frames)". Inside it show the compact arrow chain: "Latest RGB + tactile observation" → "Frozen V-JEPA encoder + Tactile encoder" → "Observed latent zobs(τi)" → "Latent comparison" → "Residual MLP" → "Δa remaining" → "+" → "Corrected remaining action". Add "same index i: predicted waypoint ẑ(τi) ↔ observed latent zobs(τi)" and write exactly "e(τi) = zobs(τi) − ẑ(τi)". Use one circular feedback arrow from this schematic back to the timeline, labeled "repeat at τ1…τ5". This one schematic stands for all five intermediate checkpoints; do not draw separate cards, a separate loop box, or a second residual module.

Inside that ONE schematic, draw exactly three arrows into "Residual MLP" from "predicted latent", "observed latent", and "latent error". Draw one arrow out labeled "Δa remaining" into a plus node. The other plus-node input is "Original action chunk a⁰ remaining". Output: "Corrected remaining action = a⁰ remaining + Δa remaining". Draw one arrow from this corrected action ONLY to gray unexecuted action slots. Gray executed slots are locked and labeled "executed actions are never modified".

At the rightmost end of the timeline show "τ6: final observation". Do not draw a residual correction arrow after τ6 because no action remains.

============================================================
ARROW AND CAUSALITY AUDIT — MUST FOLLOW EXACTLY
============================================================

Required arrows: current RGB frame → VLM; current tactile frame → VLM; current proprioceptive state → VLM; VLM → KV-cache context → Action Expert; RGB history → Frozen V-JEPA 2.1 Visual Encoder → auxiliary world model; tactile history → Trainable Tactile Encoder → auxiliary world model; proprioceptive state history → Trainable Proprioceptive State Encoder → auxiliary world model; exactly one world-model future latent plan arrow → Action Expert conditioning socket; original action chunk → execution timeline; the one representative checkpoint → latent comparison → the one Residual MLP; Residual MLP → Δa → plus node; original remaining action → plus node; plus node → remaining unexecuted slots; one feedback loop from corrected execution back to the timeline.

Forbidden arrows: history streams → VLM; action chunk → world model; world model → VLM; future latent plan → VLM or KV-cache; world model → action output directly; any second latent-guidance arrow; residual MLP → VLM or Action Expert; any arrow that regenerates the action chunk; τ6 → correction; any mismatched τ comparison; any extra decorative arrow.

============================================================
TRAINING / INFERENCE CONVENTIONS
============================================================

Use dashed outlines and badges:
"TRAINING ONLY: random seed + random 6 time offsets"
"INFERENCE: explicit offsets; latent guidance to Action Expert"
Solid modules are deployed. Training-only sampling must not look like an inference sensor or action input.

At the absolute bottom of the canvas, below every module and below the compact execution panel, add one slim full-width horizontal legend bar with a thin dark outline and minimal rounding, matching the reference paper-figure style. Use a small flat colored rectangular swatch followed by an exact explanation label. Include these entries from left to right:
"Blue = RGB / V-JEPA visual stream"
"Green = tactile stream"
"Purple = proprioceptive state"
"Orange = Action Expert"
"Red = online latent error and residual correction"
"Gray = frozen or already executed components"
"Dashed border = training-only or inference-only annotation"
Keep the legend compact, evenly spaced, and legible; do not place it inside the execution panel and do not omit any swatch.

Final quality constraint: the VLM and Action Expert visually dominate the composition; the three world-model encoders and the world model are clearly auxiliary. Leave intentional white space rather than stretching every panel to fill the canvas. The VLM box must remain visually empty except for its label and the three current-observation input arrows. Prioritize exact arrow topology and exact τ-index labels over decorative detail. Make it look like a real robotics paper figure, not a generic flowchart.
```

生成后必须人工检查：

1. 世界模型位于 VLM/Action Expert 上方且明显更小。
2. VLM 只收到当前单帧 RGB、当前单帧触觉和当前本体状态；4 帧历史只进入世界模型。世界模型分支必须明确画出 Frozen V-JEPA 2.1 Visual Encoder、Trainable Tactile Encoder、Trainable Proprioceptive State Encoder。
3. 世界模型没有收到 action chunk，只有一条 latent guidance 箭头进入 Action Expert conditioning socket。
4. VLM 通过 KV-cache 将当前上下文传给 Action Expert；Action Expert 另外接收 Future latent plan。
5. 原始 action chunk 只有一条箭头进入执行时间线，并标注只生成一次。
6. Residual MLP 只修改 remaining action，不回到 VLM/Action Expert，也不重新生成 action chunk。
7. 闭环修正只画一个紧凑的代表性示意，不要出现三个底部子面板、独立 execute-correct loop 或五个重复卡片；用 τi 表示每 5 帧重复。τ6 只有 final observation，没有 correction。
8. random seed/random offsets 使用虚线训练标注，deployment offsets 使用虚线推理标注。
