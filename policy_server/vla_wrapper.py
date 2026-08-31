"""Policy backends. Start here: the dummy policy validates shapes, transport,
and scheduling without downloading any model weights. A real backend (LeRobot,
OpenPI) implements the same `PolicyBackend` interface -- see the README's
"Choose a policy backend" section for why those integrate as separately
launched adapters rather than merged dependencies.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class PolicyBackend(Protocol):
    action_dim: int
    horizon: int

    def predict(self, observation: dict[str, Any]) -> np.ndarray:
        """Return an (horizon, action_dim) float32 array of actions."""
        ...


class DummyPolicy:
    """Ignores the observation; returns small bounded random actions.

    Exists to exercise the transport/scheduling/shape-validation path end to
    end before any real model is involved -- see README "Choose a policy
    backend".
    """

    def __init__(self, action_dim: int, horizon: int, seed: int | None = None) -> None:
        self.action_dim = action_dim
        self.horizon = horizon
        self._rng = np.random.default_rng(seed)

    def predict(self, observation: dict[str, Any]) -> np.ndarray:
        del observation  # unused: the dummy policy doesn't look at it
        return self._rng.uniform(
            -0.05, 0.05, size=(self.horizon, self.action_dim)
        ).astype(np.float32)
