"""Visual preprocessing and frozen V-JEPA runtime helpers."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def _image_tensor_to_float(images: torch.Tensor) -> torch.Tensor:
    """Convert decoded uint8 frames on the accelerator."""
    if images.dtype == torch.uint8:
        return images.float().div_(255.0)
    images = images.float()
    if images.numel() and float(images.max()) > 1.5:
        images.div_(255.0)
    return images

def _prepare_visual_gpu(
    video: torch.Tensor,
    size: int,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    video = _image_tensor_to_float(video)
    if video.shape[-2:] != (size, size):
        batch, time, channels, height, width = video.shape
        flat = video.reshape(batch * time, channels, height, width)
        flat = F.interpolate(
            flat, size=(size, size), mode="bilinear", align_corners=False
        )
        video = flat.view(batch, time, channels, size, size)
    mean_t = torch.tensor(
        mean, device=video.device, dtype=video.dtype
    ).view(1, 1, -1, 1, 1)
    std_t = torch.tensor(
        std, device=video.device, dtype=video.dtype
    ).view(1, 1, -1, 1, 1)
    return (video - mean_t) / std_t

def _visual_tokens(z, batch: int, timesteps: int):
    while isinstance(z, (list, tuple)):
        if not z:
            raise ValueError("visual encoder returned an empty output")
        z = z[-1]
    if z.ndim == 3:
        if z.shape[0] != batch:
            raise ValueError("visual latent batch dimension does not match input")
        if z.shape[1] % timesteps:
            if (z.shape[1] - 1) % timesteps == 0:
                z = z[:, 1:]
            else:
                raise ValueError(
                    "visual latent cannot be reshaped into timestep tokens"
                )
        z = z.reshape(batch, timesteps, z.shape[1] // timesteps, z.shape[2])
    elif z.ndim != 4:
        raise ValueError(
            f"visual latent must be [B,N,D] or [B,T,N,D], got {tuple(z.shape)}"
        )
    if z.shape[0] != batch or z.shape[1] != timesteps:
        raise ValueError(
            f"visual latent time shape {tuple(z.shape[:2])} does not match "
            f"(batch,timesteps)=({batch},{timesteps})"
        )
    return z

def _run_visual(encoder, video):
    if hasattr(encoder, "backbone"):
        return encoder([video], masks=None, training_mode=False)
    return encoder(video, training=False)
