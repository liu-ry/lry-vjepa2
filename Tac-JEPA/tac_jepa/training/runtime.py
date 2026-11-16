"""Distributed runtime, checkpoint directory, scheduler, and EMA helpers."""

from __future__ import annotations

import atexit
import math
import os
import re
from pathlib import Path

import torch
import torch.distributed as dist

_OUTPUT_RUN_SUFFIX = re.compile(r"^(?P<stem>.*)\((?P<n>\d+)\)$")


def _cuda_unavailable_reason() -> str:
    parts = [f"torch={torch.__version__}", f"compiled_cuda={torch.version.cuda}"]
    try:
        torch.cuda.init()
    except Exception as exc:
        text = str(exc).strip()
        if text:
            parts.append(text)
    return " ".join(parts)

def resolve_device(requested=None) -> str:
    """Resolve an explicit CPU or CUDA device without silent fallback."""
    requested = "cuda" if requested is None else str(requested).strip()
    if not requested:
        requested = "cuda"
    kind = requested.lower()
    if kind == "cpu":
        return "cpu"
    if kind == "cuda" or kind.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available to this PyTorch process; "
                "refusing to fall back to CPU. "
                f"{_cuda_unavailable_reason()} "
                "Install a wheel that matches the NVIDIA driver, or set device: cpu."
            )
        if kind == "cuda":
            return "cuda"
        suffix = kind.split(":", 1)[1]
        if not suffix.isdigit():
            raise ValueError(
                f"device must be 'cpu', 'cuda', or 'cuda:N', got {requested!r}"
            )
        index = int(suffix)
        count = torch.cuda.device_count()
        if index < 0 or index >= count:
            raise RuntimeError(
                f"device {requested} is out of range for {count} visible GPU(s)"
            )
        return f"cuda:{index}"
    raise ValueError(
        f"device must be 'cpu', 'cuda', or 'cuda:N', got {requested!r}"
    )

def setup_distributed(requested_device=None):
    """Initialize one NCCL process group when launched through torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size <= 1:
        return 0, 1, resolve_device(requested_device)
    if not torch.cuda.is_available():
        raise RuntimeError("distributed training requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only "
            f"{torch.cuda.device_count()} CUDA devices exist"
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    atexit.register(dist.destroy_process_group)
    if dist.get_world_size() != world_size or dist.get_rank() != rank:
        raise RuntimeError("torch.distributed environment is inconsistent")
    return rank, world_size, f"cuda:{local_rank}"

def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def distributed_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])

def _output_stem(name: str) -> tuple[str, int]:
    match = _OUTPUT_RUN_SUFFIX.match(name)
    if match:
        return match.group("stem"), int(match.group("n"))
    return name, 0

def _existing_output_indices(parent: Path, stem: str) -> list[int]:
    if not parent.is_dir():
        return []
    indices = []
    if (parent / stem).exists():
        indices.append(0)
    for child in parent.iterdir():
        match = _OUTPUT_RUN_SUFFIX.match(child.name)
        if match and match.group("stem") == stem:
            indices.append(int(match.group("n")))
    return indices

def unique_output_dir(path, resume=None) -> str:
    """Return a new run directory, unless resuming into the same run."""
    path = Path(path)
    parent = path.parent
    stem, _ = _output_stem(path.name)
    if resume:
        resume_dir = Path(resume).resolve().parent
        resume_stem, _ = _output_stem(resume_dir.name)
        if resume_stem == stem and resume_dir.parent.resolve() == parent.resolve():
            return str(resume_dir)
    existing = _existing_output_indices(parent, stem)
    if not existing:
        return str(path)
    return str(parent / f"{stem}({max(existing) + 1})")

def ensure_writable_output(
    path, protected_roots=("/data/upload/recorder_1",)
) -> str:
    target = Path(path).resolve()
    for root in protected_roots:
        protected = Path(root).resolve()
        if target == protected or target.is_relative_to(protected):
            raise ValueError(
                f"output {target} is inside protected raw-data root {protected}; "
                "write checkpoints and reports outside the source dataset"
            )
    return str(path)

class WarmupCosineMultiplier:
    """Linear warmup followed by cosine decay for each parameter group."""

    def __init__(
        self,
        optimizer,
        total_steps,
        warmup_steps=0,
        start_lr_scale=0.01,
        min_lr_scale=0.01,
    ):
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if not 0.0 <= start_lr_scale <= 1.0:
            raise ValueError("start_lr_scale must be in [0, 1]")
        if not 0.0 <= min_lr_scale <= 1.0:
            raise ValueError("min_lr_scale must be in [0, 1]")
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = min(int(warmup_steps), self.total_steps)
        self.start_lr_scale = float(start_lr_scale)
        self.min_lr_scale = float(min_lr_scale)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self._step = 0
        self._apply()

    def scale(self, step):
        step = int(step)
        if self.warmup_steps > 0 and step < self.warmup_steps:
            progress = (step + 1) / float(self.warmup_steps)
            return self.start_lr_scale + progress * (1.0 - self.start_lr_scale)
        cosine_steps = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, (step - self.warmup_steps) / cosine_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_scale + (1.0 - self.min_lr_scale) * cosine

    def _apply(self):
        factor = self.scale(self._step)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor
        return factor

    def step(self):
        self._step += 1
        return self._apply()

    def jump_to(self, step):
        self._step = max(0, int(step))
        return self._apply()

    def get_last_lr(self):
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self):
        return {
            "step": self._step,
            "base_lrs": list(self.base_lrs),
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "start_lr_scale": self.start_lr_scale,
            "min_lr_scale": self.min_lr_scale,
        }

    def load_state_dict(self, state):
        self._step = int(state.get("step", 0))
        base_lrs = state.get("base_lrs")
        if base_lrs is not None and len(base_lrs) == len(self.base_lrs):
            self.base_lrs = [float(value) for value in base_lrs]
        self._apply()

def update_ema(online, target, momentum: float) -> None:
    for online_param, target_param in zip(online.parameters(), target.parameters()):
        target_param.data.mul_(momentum).add_(
            online_param.data, alpha=1.0 - momentum
        )
    for online_buffer, target_buffer in zip(online.buffers(), target.buffers()):
        target_buffer.copy_(online_buffer)
