"""基于 JSONL manifest 的同步机器人视触数据集。

每行包含视觉/触觉路径，可选本体状态：
``{"vision": "...", "tactile": "...", "state": "..."}``。
数组为 [T,C,H,W]（图像）或 [T,F]（向量）。
"""
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _load(path):
    raw = torch.load(path, map_location="cpu") if str(path).endswith((".pt", ".pth")) else np.load(path)
    raw_tensor = torch.as_tensor(raw)
    x = raw_tensor.float()
    # 相机/触觉图常见 uint8，转到 [0,1]。
    if x.ndim >= 3 and x.min() >= 0 and (not torch.is_floating_point(raw_tensor) or x.max() > 1.5):
        x = x / 255.0
    return x


class RobotVisionTactileDataset(Dataset):
    def __init__(self, manifest, context=8, horizon=4, transform=None, tactile_transform=None,
                 state_transform=None, random_crop=True, require_state=False,
                 vision_size=None, normalize_vision=True,
                 vision_mean=(0.485, 0.456, 0.406),
                 vision_std=(0.229, 0.224, 0.225)):
        self.root = Path(manifest).parent
        with open(manifest, "r", encoding="utf-8") as f:
            self.items = [json.loads(x) for x in f if x.strip()]
        if not self.items:
            raise ValueError("manifest is empty")
        if any("vision" not in r or "tactile" not in r for r in self.items):
            raise ValueError("each manifest record must contain vision and tactile paths")
        if require_state and any("state" not in r for r in self.items):
            raise ValueError("state is required: add a state path to every manifest record")
        if context <= 0 or horizon <= 0:
            raise ValueError("context and horizon must be positive")
        self.context, self.horizon = context, horizon
        self.transform, self.tactile_transform = transform, tactile_transform
        self.state_transform, self.require_state = state_transform, require_state
        self.random_crop = random_crop
        self.vision_size = vision_size
        self.normalize_vision = bool(normalize_vision)
        self.vision_mean = torch.tensor(vision_mean, dtype=torch.float32).view(1, -1, 1, 1)
        self.vision_std = torch.tensor(vision_std, dtype=torch.float32).view(1, -1, 1, 1)

    def __len__(self): return len(self.items)

    def _prepare_vision(self, vision):
        """返回符合 V-JEPA 约定的 ``[T,C,H,W]`` 浮点输入。"""
        if vision.ndim != 4:
            raise ValueError(f"vision must be a 4-D frame sequence, got {tuple(vision.shape)}")
        # manifest 约定是 TCHW。同时接受常见的 THWC / CTHW，避免导出时通道被静默打乱。
        if vision.shape[-1] in (1, 3) and vision.shape[1] > 3:
            vision = vision.permute(0, 3, 1, 2)
        elif vision.shape[0] in (1, 3) and vision.shape[1] > 3:
            vision = vision.permute(1, 0, 2, 3)
        if vision.shape[1] not in (1, 3):
            raise ValueError(f"vision must have 1 or 3 channels, got shape {tuple(vision.shape)}")
        if vision.shape[1] == 1:
            vision = vision.repeat(1, 3, 1, 1)
        if self.vision_size is not None:
            size = (int(self.vision_size), int(self.vision_size)) if isinstance(self.vision_size, int) else tuple(self.vision_size)
            if tuple(vision.shape[-2:]) != size:
                vision = F.interpolate(vision, size=size, mode="bilinear", align_corners=False)
        if self.normalize_vision:
            # 外部 transform 可能已经归一化过，避免再套一层 ImageNet mean/std。
            if float(vision.min()) >= -1e-3 and float(vision.max()) <= 1.001:
                mean = self.vision_mean.to(device=vision.device, dtype=vision.dtype)
                std = self.vision_std.to(device=vision.device, dtype=vision.dtype)
                vision = (vision - mean) / std
        return vision

    @staticmethod
    def _prepare_tactile(tactile):
        """触觉图像序列保持 ``[T,C,H,W]``。"""
        if tactile.ndim == 4 and tactile.shape[0] in (1, 2, 3) and tactile.shape[1] > 3:
            tactile = tactile.permute(1, 0, 2, 3)
        return tactile

    def __getitem__(self, i):
        r = self.items[i]
        v, t = _load(self.root / r["vision"]), _load(self.root / r["tactile"])
        s = _load(self.root / r["state"]) if "state" in r else None
        a = _load(self.root / r["action"]) if "action" in r else None
        n = self.context + self.horizon
        usable = min(
            len(v), len(t), len(s) if s is not None else len(v),
            len(a) if a is not None else len(v),
        )
        if usable < n: raise ValueError("sequence shorter than context+horizon")
        start = random.randint(0, usable - n) if self.random_crop and usable > n else 0
        v, t = v[start:start+n], t[start:start+n]
        if s is not None: s = s[start:start+n]
        if a is not None: a = a[start:start+n]
        if self.transform: v = self.transform(v)
        v = self._prepare_vision(v)
        if self.tactile_transform: t = self.tactile_transform(t)
        t = self._prepare_tactile(t)
        if self.state_transform and s is not None: s = self.state_transform(s)
        # 这里保持 [T,C,H,W]，训练循环再转成 V-JEPA 的 [B,C,T,H,W]。
        out = {"vision": v, "tactile": t, "context": self.context}
        if s is not None: out["state"] = s
        if a is not None: out["action"] = a
        return out
