# Stage A/B closure smoke test

This smoke test used 12 synthetic episodes and a deliberately tiny model. It
validated code execution, data loading, checkpoint flow, and deployment
inference. The synthetic data and generated weights are kept outside the
repository.

## Synthetic data

- 12 episodes
- 48 independent tactile-chain records
- 4 tactile chains and 2 vision/state arms per episode
- 40 frames per coordinate stream
- 34-frame `4 context + 30 horizon` training windows
- Stage A grouped split: 32 / 8 / 8 chain records
- Stage B grouped split: 8 / 2 / 2 episodes
- Stage B windows: 16 / 4 / 4

Tactile input was resized to `336×224` through the original encoder contract.
The test used `embed_dim=32`, one encoder layer, one predictor layer, and two
condition tokens.

## Stage A

Executed on CPU for two optimizer steps:

```bash
python -m tac_jepa.training.stage_a \
  --config /tmp/tac-jepa-smoke.bT36LY/stage_a.yaml \
  --output /tmp/tac-jepa-smoke.bT36LY/runs/stage_a \
  --device cpu --max-steps 2 --batch-size 2 --num-workers 0
```

Result:

- Step 1 and step 2 completed.
- Validation and test forward passes completed.
- `best.pt`, `latest.pt`, `metrics.json`, and `test_metrics.json` were written.
- Best score: `0.9209498491583886` at step 2.

## Stage B

Stage B loaded the Stage A encoder checkpoint and executed two optimizer steps:

```bash
python -m tac_jepa.training.stage_b \
  --config /tmp/tac-jepa-smoke.bT36LY/stage_b.yaml \
  --encoder-checkpoint '/tmp/tac-jepa-smoke.bT36LY/runs/stage_a(1)/best.pt' \
  --output /tmp/tac-jepa-smoke.bT36LY/runs/stage_b \
  --device cpu --attention per_frame_causal --latent-grid 2x2 \
  --max-steps 2 --batch-size 2 --num-workers 0 \
  --projection mlp --condition-tokens 2 \
  --predictor-depth 1 --condition-depth 1
```

Result:

- Tactile, visual, and state inputs were loaded through the copied datasets.
- Tactile and visual encoders ran.
- Future tactile/visual prediction, auxiliary summaries, and probes ran.
- Validation and test forward passes completed.
- `best_predictor.pt`, `latest_predictor.pt`, `metrics.json`, `split.json`,
  and `test_metrics.json` were written.
- Best score: `1.0355797421697155` at step 2.

## Inference

`TactileFutureFeatureExtractor` loaded both training products and ran a
deployment forward pass. The tiny model produced:

```text
tactile_encoder        85,280 params
visual_encoder_loaded  5,368,320 params
world_predictor        140,066 params

tactile_context:  (1, 4, 4, 4, 32)
future_tactile:   (1, 4, 3, 4, 32)
future_visual:    (1, 2, 1, 192)
condition_tokens: (1, 2, 32)
```

All floating-point outputs were finite. The Stage-B predictor output horizon
was shortened to three frames for the smoke test; production configuration
uses the original 30-frame horizon.

## Dataset preparation

Four tiny recorder-format episodes were generated with two arms, four tactile
videos, four state files, and two vision videos per episode. The unified
command completed all three processing stages:

```bash
python tools/prepare_dataset.py \
  --format recorder \
  --root /tmp/tac-jepa-smoke.bT36LY/raw_recorder \
  --output /tmp/tac-jepa-smoke.bT36LY/prepared \
  --workers 2
```

Result:

- 8 arm-level camera records converted
- 8 contact/slip records generated
- 16 chain records generated
- 4 four-chain episode records generated
- output manifests were written to `prepared/chains/`

## One-command training

The sequential training wrapper was run for one Stage-A step and one Stage-B
step. It successfully passed `stage_a/best.pt` to Stage B:

```bash
python tools/train_all.py \
  --stage-a-config /tmp/tac-jepa-smoke.bT36LY/stage_a.yaml \
  --stage-b-config /tmp/tac-jepa-smoke.bT36LY/stage_b.yaml \
  --output-root /tmp/tac-jepa-smoke.bT36LY/runs/full \
  --device cpu --latent-grid 2x2 \
  --stage-a-max-steps 1 --stage-b-max-steps 1 \
  --stage-a-batch-size 2 --stage-b-batch-size 2 \
  --num-workers 0
```

## Fixes found by the test

- Restored `_OUTPUT_RUN_SUFFIX` in `training/runtime.py`.
- Corrected the Stage-B local alias for the visual builder.
- Added `--device` to both training entry points so closure tests can run on CPU
  without occupying GPUs used by production training.
