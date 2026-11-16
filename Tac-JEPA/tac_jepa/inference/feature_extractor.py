"""Portable deployment inference for the tactile world model."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from ..models.chain_encoder import (
    ChainTactileEncoder,
    _align_temporal_embedding,
    _causal_delta_clip,
    _encode_chains,
    _resize_clip,
)
from ..models.predictor import ChainWorldPredictor
from ..vision.builder import build_visual
from ..vision.ops import (
    _prepare_visual_gpu,
    _run_visual,
    _visual_tokens,
)


def _image_size(cfg: dict) -> tuple[int, int]:
    value = cfg.get("tactile", {}).get("img_size", (336, 224))
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError("tactile.img_size must be an int or [height, width]")
    return int(value[0]), int(value[1])


def _checkpoint_payload(path: str | Path, key: str) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if key in state:
        return state[key]
    return state


class TactileFutureFeatureExtractor:
    """Load tactile encoder, visual encoder, and future latent predictor.

    This deployment path intentionally does not construct the Stage-A EMA
    target encoder, contact/slip probes, datasets, or training modules.
    """

    def __init__(
        self,
        config_path: str | Path,
        encoder_checkpoint: str | Path,
        predictor_checkpoint: str | Path,
        *,
        device: str = "cuda",
        attention: str = "per_frame_causal",
        latent_grid: tuple[int, int] = (7, 7),
        visual_checkpoint: str | Path | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.cfg = yaml.safe_load(
            self.config_path.read_text(encoding="utf-8")
        )
        self.device = device
        self.context = int(self.cfg["context"])
        self.horizon = int(self.cfg["horizon"])
        self.image_size = _image_size(self.cfg)
        self.latent_grid = tuple(int(value) for value in latent_grid)
        self.source_grid = (
            self.image_size[0] // int(self.cfg["tactile"]["patch_size"]),
            self.image_size[1] // int(self.cfg["tactile"]["patch_size"]),
        )

        tactile_cfg = self.cfg["tactile"]
        embed_dim = int(tactile_cfg.get("embed_dim", 384))
        in_channels = int(tactile_cfg.get("in_channels", 3))
        if bool(tactile_cfg.get("use_delta", True)):
            in_channels *= 2
        self.encoder = ChainTactileEncoder(
            attention_mode=attention,
            embed_dim=embed_dim,
            depth=int(tactile_cfg.get("depth", 6)),
            heads=int(tactile_cfg.get("num_heads", 6)),
            image_size=self.image_size,
            max_time=self.context + self.horizon,
            causal_depth=int(
                self.cfg.get("encoder_training", {}).get(
                    "causal_temporal_depth", 2
                )
            ),
            drop_path=float(tactile_cfg.get("drop_path", 0.1)),
        ).to(device)
        encoder_state = _checkpoint_payload(encoder_checkpoint, "encoder")
        self.encoder.load_state_dict(
            _align_temporal_embedding(
                encoder_state,
                self.encoder.state_dict(),
            )
        )
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

        predictor_cfg = self.cfg["predictor_training"]
        if visual_checkpoint is not None:
            predictor_cfg = dict(predictor_cfg)
            predictor_cfg["vjepa_checkpoint"] = str(visual_checkpoint)
        self.visual = build_visual(self.cfg, predictor_cfg, device)
        self.state_dim = int(
            predictor_cfg.get(
                "state_history_dim",
                self.cfg.get("state_history_dim", 8),
            )
        )
        self.predictor = ChainWorldPredictor(
            tactile_dim=embed_dim,
            visual_dim=int(self.visual.embed_dim),
            state_dim=self.state_dim,
            spatial_tokens=self.latent_grid[0] * self.latent_grid[1],
            hidden_dim=int(predictor_cfg.get("predictor_dim", 384)),
            depth=int(predictor_cfg.get("predictor_depth", 2)),
            condition_depth=int(predictor_cfg.get("condition_depth", 2)),
            heads=int(predictor_cfg.get("predictor_heads", 6)),
            max_time=self.context + self.horizon + 1,
            future_offsets=tuple(
                int(value)
                for value in self.cfg.get(
                    "future_offsets",
                    range(1, self.horizon + 1),
                )
            ),
            visual_future_offsets=tuple(
                int(value)
                for value in self.cfg.get(
                    "visual_future_offsets",
                    range(2, self.horizon + 1, 2),
                )
            ),
            num_condition_tokens=int(
                predictor_cfg.get("num_condition_tokens", 4)
            ),
            projection_type=str(
                predictor_cfg.get("modality_projection", "linear")
            ),
            drop_path=float(predictor_cfg.get("drop_path", 0.05)),
        ).to(device)
        predictor_state = _checkpoint_payload(
            predictor_checkpoint, "predictor"
        )
        self.predictor.load_state_dict(predictor_state)
        self.predictor.eval()

        self.state_scale = torch.as_tensor(
            self.cfg.get("state_scale", [1.0] * self.state_dim),
            device=device,
            dtype=torch.float32,
        ).view(1, 1, 1, -1)
        if self.state_scale.shape[-1] != self.state_dim:
            raise ValueError("state_scale width does not match state_dim")

    def _prepare_tactile(
        self,
        tactile_chains: torch.Tensor,
    ) -> torch.Tensor:
        chains = tactile_chains.to(self.device, non_blocking=True)
        if chains.dtype == torch.uint8:
            chains = chains.float().div_(255.0)
        else:
            chains = chains.float()
        batch, count, time, channels, height, width = chains.shape
        flat = chains.reshape(
            batch * count, time, channels, height, width
        )
        flat = _resize_clip(flat, self.image_size)
        flat = _causal_delta_clip(flat)
        return flat.reshape(
            batch,
            count,
            time,
            flat.shape[2],
            *self.image_size,
        )

    def _visual_context(self, vision_arms: torch.Tensor) -> torch.Tensor:
        if vision_arms.shape[2] != self.context:
            raise ValueError(
                f"vision_arms must contain exactly {self.context} frames"
            )
        video = vision_arms.to(self.device, non_blocking=True).float()
        batch, arms, time, channels, height, width = video.shape
        tubelet = int(self.cfg.get("visual_tubelet_size", 2))
        steps = time // tubelet
        flat = _prepare_visual_gpu(
            video.reshape(
                batch * arms,
                time,
                channels,
                height,
                width,
            ),
            int(self.cfg.get("crop_size", 384)),
        ).permute(0, 2, 1, 3, 4).contiguous()
        with torch.no_grad():
            tokens = _visual_tokens(
                _run_visual(self.visual, flat),
                batch * arms,
                steps,
            )
        tokens = tokens.reshape(
            batch,
            arms,
            steps,
            tokens.shape[-2],
            tokens.shape[-1],
        )
        return tokens.mean(dim=-2)

    @torch.no_grad()
    def __call__(
        self,
        tactile_chains: torch.Tensor,
        vision_arms: torch.Tensor,
        state_arms: torch.Tensor,
        *,
        chain_valid: torch.Tensor | None = None,
        arm_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return tactile context, future latents, and query tokens."""
        if tactile_chains.shape[2] != self.context:
            raise ValueError(
                f"tactile_chains must contain exactly {self.context} frames"
            )
        if state_arms.shape[2] != self.context:
            raise ValueError(
                f"state_arms must contain exactly {self.context} frames"
            )
        batch = tactile_chains.shape[0]
        if chain_valid is None:
            chain_valid = torch.ones(
                batch,
                self.predictor.num_chains,
                dtype=torch.bool,
                device=self.device,
            )
        else:
            chain_valid = chain_valid.to(self.device, dtype=torch.bool)
        if arm_valid is None:
            arm_valid = torch.stack(
                (
                    chain_valid[:, 0] | chain_valid[:, 1],
                    chain_valid[:, 2] | chain_valid[:, 3],
                ),
                dim=1,
            )
        else:
            arm_valid = arm_valid.to(self.device, dtype=torch.bool)

        clip = self._prepare_tactile(tactile_chains)
        context_indices = list(range(self.context))
        tactile_context = _encode_chains(
            self.encoder,
            clip,
            context_indices,
            self.source_grid,
            self.latent_grid,
        )
        visual_context = self._visual_context(vision_arms)
        state_history = (
            state_arms.to(self.device, non_blocking=True).float()
            / self.state_scale
        )
        output = self.predictor(
            tactile_context,
            context_indices,
            chain_valid,
            visual_context,
            list(range(visual_context.shape[2])),
            arm_valid,
            state_history,
            self.context,
        )
        return {
            "tactile_context": tactile_context,
            "future_tactile": output["future_tactile"],
            "future_visual": output["future_visual"],
            "condition_tokens": output["condition_tokens"],
            "chain_valid": chain_valid,
            "arm_valid": arm_valid,
        }

    def parameter_counts(self) -> dict[str, int]:
        return {
            "tactile_encoder": sum(
                parameter.numel() for parameter in self.encoder.parameters()
            ),
            "visual_encoder_loaded": sum(
                parameter.numel() for parameter in self.visual.parameters()
            ),
            "world_predictor": sum(
                parameter.numel() for parameter in self.predictor.parameters()
            ),
        }


__all__ = ["TactileFutureFeatureExtractor"]
