"""Standalone inference interface for the tactile encoder and future predictor."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from app.vjepa_2_1.tactile_training_utils import (
    _run_visual,
    _visual_tokens,
)
from app.vjepa_2_1.train_tactile_chain_stage_b import (
    ChainStageB,
    _encode_chains,
)
from app.vjepa_2_1.train_tactile_two_stage import _prepare_visual_gpu


class TactileFutureFeatureExtractor:
    """Load the final tactile encoder/predictor and expose downstream features.

    Inputs are the four-frame context used by Stage B. The returned query
    tokens and future latents can be consumed by a separate downstream model.
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
    ) -> None:
        self.config_path = Path(config_path)
        self.cfg = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.device = device
        self.runner = ChainStageB(
            self.cfg,
            device,
            str(encoder_checkpoint),
            attention=attention,
            latent_grid=latent_grid,
        )
        state = torch.load(
            predictor_checkpoint, map_location=device, weights_only=False
        )
        for name in ("predictor", "contact_probe", "slip_probe"):
            getattr(self.runner, name).load_state_dict(state[name])
        for module in (
            self.runner.predictor,
            self.runner.contact_probe,
            self.runner.slip_probe,
        ):
            module.eval()

    def _visual_context(self, vision_arms: torch.Tensor) -> torch.Tensor:
        video = vision_arms.to(self.device, non_blocking=True).float()
        batch, arms, time, channels, height, width = video.shape
        time = min(time, self.runner.context)
        if time != self.runner.context:
            raise ValueError(
                f"at least {self.runner.context} visual frames are required"
            )
        tubelet = int(self.cfg.get("visual_tubelet_size", 2))
        steps = time // tubelet
        if steps <= 0:
            raise ValueError("visual context is shorter than one tubelet")
        flat = _prepare_visual_gpu(
            video.reshape(batch * arms, time, channels, height, width),
            int(self.cfg.get("crop_size", 384)),
        ).permute(0, 2, 1, 3, 4).contiguous()
        with torch.no_grad():
            tokens = _visual_tokens(
                _run_visual(self.runner.visual, flat),
                batch * arms,
                steps,
            )
        tokens = tokens.reshape(
            batch, arms, steps, tokens.shape[-2], tokens.shape[-1]
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
        """Return Stage-B context, future latents, and query tokens.

        Args:
            tactile_chains: ``[B,4,T,3,H,W]`` raw tactile frames.
            vision_arms: ``[B,2,T,3,H,W]`` visual frames.
            state_arms: ``[B,2,T,state_dim]`` state history.
            chain_valid: Optional ``[B,4]`` tactile-chain mask.
            arm_valid: Optional ``[B,2]`` active-arm mask.
        """
        batch = tactile_chains.shape[0]
        if chain_valid is None:
            chain_valid = torch.ones(
                batch, 4, dtype=torch.bool, device=self.device
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

        clip, _ = self.runner._prepare_tactile(
            {"tactile_chains": tactile_chains}
        )
        context_indices = list(range(self.runner.context))
        tactile_context = _encode_chains(
            self.runner.encoder,
            clip,
            context_indices,
            self.runner.source_grid,
            self.runner.latent_grid,
        )
        visual_context = self._visual_context(vision_arms)
        state_history = self.runner._state_history({"state_arms": state_arms})
        output = self.runner.predictor(
            tactile_context,
            context_indices,
            chain_valid,
            visual_context,
            list(range(visual_context.shape[2])),
            arm_valid,
            state_history,
            self.runner.context,
        )
        return {
            "tactile_context": tactile_context,
            "future_tactile": output["future_tactile"],
            "future_visual": output["future_visual"],
            "condition_tokens": output["condition_tokens"],
            "chain_valid": chain_valid,
            "arm_valid": arm_valid,
        }


__all__ = ["TactileFutureFeatureExtractor"]
