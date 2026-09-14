"""Adapter for a real OpenPI checkpoint served over OpenPI's own protocol.

This calls the actual, separately-installed `openpi-client` PyPI package
(https://pypi.org/project/openpi-client/) -- synchronous request-response
over WebSocket + msgpack. It does not reimplement, vendor, or copy any of
OpenPI's serving or model code; see
.reference/openpi-main/packages/openpi-client (not committed -- see README's
"License and external assets") for the reference this was written against.

Install: `pip install openpi-client` into its OWN environment, separate from
this project's core `.venv` -- openpi-client pins `numpy<2.0.0`, which
conflicts with the numpy 2.x this project's torch/mujoco need. Run this
adapter (and policy_server generally) from that separate environment when
using the openpi backend, the same way the README already tells you to keep
a from-source LeRobot/OpenPI checkout out of the core environment.

You also need a real OpenPI policy server already running and reachable at
`host:port` -- e.g. via OpenPI's own `scripts/serve_policy.py` from a real
checkpoint. This project supplies neither a checkpoint nor OpenPI's model
code; see OpenPI's own README for how to obtain and serve one.

Architectural note (see also the README's "Choose a policy backend" section):
OpenPI's own reference client (`ActionChunkBroker`) is a *blocking*
request-response call amortized over an action chunk -- the robot's control
loop stalls on `websocket.recv()` whenever the chunk runs out. This
project's transport (sim/zmq_publisher.py) is fully async instead: physics
never blocks on this adapter's `predict()` call, however long it takes --
the sim just keeps playing the last chunk (or falls back to holding
position) until a fresh one is ready. Wrapping OpenPI's synchronous client
this way doesn't change that; predict() runs on the policy server's own
thread (see policy_server/server.py), and the sim finds out only once it's
done.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from openpi_client import websocket_client_policy as _websocket_client_policy
except ImportError as exc:  # pragma: no cover - exercised only with the openpi extra installed
    raise ImportError(
        "OpenPIPolicy requires the 'openpi-client' package, installed in its own "
        "environment separate from this project's core .venv (see this module's "
        "docstring for why): pip install openpi-client"
    ) from exc


class OpenPIPolicy:
    """Talks to a real, separately-launched OpenPI websocket policy server.

    This project's Observation (one wrist camera, an N-joint arm, no
    gripper) doesn't match any published OpenPI checkpoint's expected input
    exactly -- real checkpoints (see droid_policy.py's make_droid_example)
    expect e.g. a 7-DoF arm + gripper + two named cameras. `image_key`/
    `state_key`/`prompt_key`/`action_key` make that mapping configurable
    rather than hardcoded, since it's checkpoint-specific, not something
    this adapter can know in general -- set them to match whatever the
    served checkpoint actually expects.
    """

    obs_horizon = 1

    def __init__(
        self,
        host: str,
        port: int,
        action_dim: int,
        horizon: int,
        image_key: str = "observation/wrist_image_left",
        state_key: str = "observation/joint_position",
        prompt_key: str = "prompt",
        action_key: str = "actions",
    ) -> None:
        self.action_dim = action_dim
        self.horizon = horizon
        self._image_key = image_key
        self._state_key = state_key
        self._prompt_key = prompt_key
        self._action_key = action_key
        self._client = _websocket_client_policy.WebsocketClientPolicy(host=host, port=port)

    def predict(self, observation: dict[str, Any]) -> np.ndarray:
        # `observation` is a stacked history (see observation_history.py);
        # obs_horizon=1 means only the newest frame is meaningful here.
        obs = {
            self._image_key: np.asarray(observation["rgb"][-1], dtype=np.uint8),
            self._state_key: np.asarray(observation["joint_positions"][-1], dtype=np.float32),
            self._prompt_key: observation.get("task_instruction", ""),
        }
        result = self._client.infer(obs)
        actions = np.asarray(result[self._action_key], dtype=np.float32)
        return actions.reshape(self.horizon, self.action_dim)
