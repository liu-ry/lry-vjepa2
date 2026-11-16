"""Build and load the frozen V-JEPA2.1 visual encoder used by Stage B."""

from __future__ import annotations

import warnings

import torch

from . import vision_transformer
from .wrapper import MultiSeqWrapper


def _load_checkpoint(
    module,
    path,
    keys=("ema_encoder", "encoder", "target_encoder"),
):
    if not path:
        return
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    for key in keys:
        if key in state:
            state = state[key]
            break
    state = {
        key.replace("module.", "", 1): value
        for key, value in state.items()
    }
    if (
        hasattr(module, "backbone")
        and state
        and not any(key.startswith("backbone.") for key in state)
    ):
        state = {"backbone." + key: value for key, value in state.items()}
    incompatible = module.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        warnings.warn(
            "visual checkpoint mismatch: "
            f"missing={list(incompatible.missing_keys)[:12]}, "
            f"unexpected={list(incompatible.unexpected_keys)[:12]}",
            stacklevel=2,
        )


def build_visual(cfg: dict, section: dict, device: str):
    model_cfg = cfg.get("model", {})
    context = int(cfg["context"])
    horizon = int(cfg["horizon"])
    encoder = vision_transformer.__dict__[
        model_cfg.get("model_name", "vit_base")
    ](
        img_size=int(cfg.get("crop_size", 384)),
        patch_size=int(model_cfg.get("patch_size", 16)),
        num_frames=context + horizon,
        tubelet_size=int(cfg.get("visual_tubelet_size", 2)),
        uniform_power=False,
        use_sdpa=bool(model_cfg.get("use_sdpa", True)),
        use_silu=False,
        wide_silu=False,
        use_activation_checkpointing=False,
        is_causal=False,
        use_rope=bool(model_cfg.get("use_rope", True)),
        init_type="default",
        img_temporal_dim_size=int(
            model_cfg.get("img_temporal_dim_size", 1)
        ),
        n_registers=0,
        has_cls_first=False,
        interpolate_rope=bool(model_cfg.get("interpolate_rope", True)),
        modality_embedding=True,
    )
    encoder = MultiSeqWrapper(encoder).to(device)
    _load_checkpoint(encoder, section["vjepa_checkpoint"])
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder
