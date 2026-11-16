"""Minimal wrapper that adapts the V-JEPA visual encoder to list inputs."""

from __future__ import annotations

import torch.nn as nn


class MultiSeqWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.embed_dim

    def forward(self, x, masks=None, gram_mode=False, training_mode=False):
        if masks is None:
            return [
                self.backbone(sequence, training=training_mode)
                for sequence in x
            ]
        outputs = [[] for _ in x]
        for index, (sequence, sequence_masks) in enumerate(zip(x, masks)):
            outputs[index].extend(
                self.backbone(
                    sequence,
                    masks=mask,
                    training=training_mode,
                )
                for mask in sequence_masks
            )
        return outputs
