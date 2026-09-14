"""Observation/action normalization.

Necessary scaffolding for any real trained policy: a model trained on
normalized inputs/outputs will perform badly, not just suboptimally, if
served raw un-normalized data at inference time -- this isn't an
optimization, it's a correctness requirement. The shape of this interface
(fit a per-dimension affine transform once, apply/invert it forever after)
mirrors diffusion_policy's LinearNormalizer
(.reference/diffusion_policy-main/diffusion_policy/model/common/normalizer.py,
not committed -- see README's "License and external assets"; not copied
from, just the same well-established idea). A real checkpoint (LeRobot,
OpenPI) ships its own fitted statistics baked in and normalizes internally,
so this class exists for this project's *own* future policies, not to
re-normalize data already handled by a real checkpoint's own pipeline.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class NormalizerLike(Protocol):
    def normalize(self, x: np.ndarray) -> np.ndarray: ...
    def denormalize(self, x: np.ndarray) -> np.ndarray: ...


class Normalizer:
    """Per-dimension affine normalization: (x - mean) / std, and its inverse."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-6) -> None:
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        if mean.shape != std.shape:
            raise ValueError(f"mean/std shape mismatch: {mean.shape} vs {std.shape}")
        self.mean = mean
        self.std = np.maximum(std, eps)  # avoid dividing by ~0 for a constant dimension

    @classmethod
    def fit(cls, data: np.ndarray) -> Normalizer:
        """`data`: (N, D) samples stacked along axis 0."""
        data = np.asarray(data, dtype=np.float32)
        return cls(mean=data.mean(axis=0), std=data.std(axis=0))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) * self.std + self.mean).astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, list[float]]) -> Normalizer:
        return cls(mean=np.array(data["mean"]), std=np.array(data["std"]))


class IdentityNormalizer:
    """No-op stand-in with the same interface as Normalizer.

    Used where real fitted statistics aren't available yet (every backend in
    this project today) so the surrounding code -- ObservationHistory, a
    future policy -- can unconditionally call `.normalize()`/`.denormalize()`
    without a None-check, and swapping in a real Normalizer later needs no
    call-site changes.
    """

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32)
