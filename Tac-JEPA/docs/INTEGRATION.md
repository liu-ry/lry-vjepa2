# Host integration contract

The portable module exposes two integration levels.

## 1. Inference only

Use `tac_jepa.TactileFutureFeatureExtractor`. It builds only:

1. causal tactile encoder
2. frozen visual encoder
3. future tactile latent predictor

It does not build the training target encoder, datasets, probes, optimizer, or
training loop.

```python
from tac_jepa import TactileFutureFeatureExtractor

model = TactileFutureFeatureExtractor(
    "configs/stage_b.example.yaml",
    "checkpoints/stage_a_encoder.pt",
    "checkpoints/stage_b_predictor.pt",
    device="cuda",
)

outputs = model(
    tactile_chains,  # [B, 4, 4, 3, H, W]
    vision_arms,     # [B, 2, 4, 3, H, W]
    state_arms,      # [B, 2, 4, 8]
)
```

The returned deployment features are:

- `tactile_context`: `[B, 4, 4, 49, 384]`
- `future_tactile`: `[B, 4, 30, 49, 384]`
- `future_visual`: `[B, 2, 15, 768]`
- `condition_tokens`: `[B, 4, 384]`

There is no action input, action head, or action output.

## 2. Training

Run Stage A before Stage B:

```bash
export PYTHONPATH=/path/to/Tac-JEPA
python -m tac_jepa.training.stage_a \
  --config configs/stage_a.example.yaml \
  --output runs/stage_a

python -m tac_jepa.training.stage_b \
  --config configs/stage_b.example.yaml \
  --encoder-checkpoint runs/stage_a/best.pt \
  --attention per_frame_causal \
  --latent-grid 7x7 \
  --output runs/stage_b
```

The host can replace `tac_jepa.vision.builder.build_visual` with another
frozen visual encoder as long as it preserves the `[B, arms, time, dim]`
context contract and V-JEPA token layout expected by Stage B.

## Data contract

The unified processing command is:

```bash
python tools/prepare_dataset.py \
  --format recorder \
  --root /data/upload/recorder_1 \
  --output data/prepared \
  --workers 16
```

It runs the three lower-level tools:

1. `tools/convert_data.py`
2. `tools/build_contacts.py`
3. `tools/build_chains.py`

The resulting training manifests are
`data/prepared/chains/chains.jsonl` and
`data/prepared/chains/episodes_4chain.jsonl`.

## Stage-B window policy

The portable Stage-B trainer separates contact supervision from window
eligibility:

- Stage A remains contact-only.
- Stage B defaults to `require_contact: false` and `window_sampling: all`.
- A no-contact Stage-B window remains valid when its tactile stream exists; the
  manifest `chain_valid` flags, not the current contact count, define the tactile
  loss mask.
- `train_window_sampling`, `val_window_sampling`, and
  `test_window_sampling` independently support `contact`, `all`, and
  `stratified`. Dense training also supports `mixed` with
  `no_contact_fraction`.
- Fixed-size validation/test views can be deterministically shuffled with
  `shuffle_eval_windows: true`.
- Adjacent windows overlap by `window_overlap: 0.3333333333333333` by default.

The example Stage-B configuration uses natural/all training windows,
contact-only Stage A, deterministic stratified evaluation, and no action input
or action output.

Data and checkpoints are intentionally not bundled. The copied module contains
only code, configuration templates, tools, and tests.
