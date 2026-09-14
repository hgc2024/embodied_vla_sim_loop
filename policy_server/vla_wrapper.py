"""Policy backends. Start here: the dummy policy validates shapes, transport,
and scheduling without downloading any model weights. A real backend (LeRobot,
OpenPI) implements the same `PolicyBackend` interface -- see the README's
"Choose a policy backend" section for why those integrate as separately
launched adapters rather than merged dependencies, and
policy_server/adapters/openpi_policy.py for a real one.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class PolicyBackend(Protocol):
    action_dim: int
    horizon: int
    # How many past observations this backend wants stacked into each
    # `predict()` call -- see policy_server/observation_history.py. 1 means
    # "just the current frame", matching every backend implemented so far;
    # a policy trained on multi-frame context (see diffusion_policy's
    # n_obs_steps) would declare more.
    obs_horizon: int

    def predict(self, observation: dict[str, Any]) -> np.ndarray:
        """`observation` is a *stacked* history from ObservationHistory:
        array fields have a leading (obs_horizon, ...) time axis. Returns an
        (horizon, action_dim) float32 array of actions."""
        ...


class DummyPolicy:
    """Ignores the observation; returns small bounded random actions.

    Exists to exercise the transport/scheduling/shape-validation path end to
    end before any real model is involved -- see README "Choose a policy
    backend".
    """

    obs_horizon = 1

    def __init__(self, action_dim: int, horizon: int, seed: int | None = None) -> None:
        self.action_dim = action_dim
        self.horizon = horizon
        self._rng = np.random.default_rng(seed)

    def predict(self, observation: dict[str, Any]) -> np.ndarray:
        del observation  # unused: the dummy policy doesn't look at it
        return self._rng.uniform(
            -0.05, 0.05, size=(self.horizon, self.action_dim)
        ).astype(np.float32)
