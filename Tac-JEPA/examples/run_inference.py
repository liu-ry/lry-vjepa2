"""Minimal inference contract for the portable tactile world model."""

from __future__ import annotations

import argparse

import torch

from tac_jepa import TactileFutureFeatureExtractor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--predictor-checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model = TactileFutureFeatureExtractor(
        args.config,
        args.encoder_checkpoint,
        args.predictor_checkpoint,
        device=args.device,
    )
    tactile = torch.rand(1, 4, 4, 3, 240, 240)
    vision = torch.rand(1, 2, 4, 3, 640, 480)
    state = torch.randn(1, 2, 4, 8)
    output = model(tactile, vision, state)
    print(model.parameter_counts())
    for name, value in output.items():
        print(name, tuple(value.shape), value.dtype)


if __name__ == "__main__":
    main()
