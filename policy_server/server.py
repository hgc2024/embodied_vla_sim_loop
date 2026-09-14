"""Policy process: subscribes to observations, publishes action chunks.

Runs the socket receive and the inference call on separate threads so a slow
or stalled inference call can never block draining the observation socket --
the buffer always holds the newest frame, and a new inference call always
starts from whatever is newest when it's ready, never from a backlog.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from policy_server.observation_history import ObservationHistory
from policy_server.ring_buffer import LatestObservationBuffer
from policy_server.vla_wrapper import DummyPolicy, PolicyBackend
from sim.zmq_publisher import ActionChunkPublisher, ObservationSubscriber, now_us

logger = logging.getLogger("policy_server")


def build_policy(config: dict[str, Any]) -> PolicyBackend:
    policy_cfg = config["policy"]
    backend = policy_cfg.get("backend", "dummy")

    if backend == "dummy":
        return DummyPolicy(action_dim=policy_cfg["action_dim"], horizon=policy_cfg["action_horizon"])

    if backend == "openpi":
        # Imported lazily: openpi_policy.py itself raises a clear ImportError
        # if openpi-client isn't installed, and importing it eagerly would
        # make every other backend (dummy included) require that optional,
        # separately-environed dependency just to start up.
        from policy_server.adapters.openpi_policy import OpenPIPolicy

        openpi_cfg = policy_cfg.get("openpi", {})
        return OpenPIPolicy(
            host=openpi_cfg["host"],
            port=openpi_cfg["port"],
            action_dim=policy_cfg["action_dim"],
            horizon=policy_cfg["action_horizon"],
            **{k: v for k, v in openpi_cfg.items() if k in ("image_key", "state_key", "prompt_key", "action_key")},
        )

    raise NotImplementedError(
        f"policy backend '{backend}' is not implemented yet; 'dummy' and 'openpi' are "
        "available (see README milestones for LeRobot)"
    )


class PolicyServer:
    def __init__(self, config: dict[str, Any]) -> None:
        cam_cfg = config["camera"]
        ipc_cfg = config["ipc"]

        self.obs_sub = ObservationSubscriber(
            ipc_cfg["observations"],
            height=cam_cfg["height"],
            width=cam_cfg["width"],
            depth_dtype=np.dtype(cam_cfg["depth_dtype"]),
            high_water_mark=ipc_cfg["high_water_mark"],
        )
        self.action_pub = ActionChunkPublisher(
            ipc_cfg["actions"], high_water_mark=ipc_cfg["high_water_mark"]
        )
        self.buffer = LatestObservationBuffer()
        self.policy = build_policy(config)
        # Frames the transport actually delivered accumulate here so the
        # policy can condition on more than just the single newest one --
        # see observation_history.py's docstring for why this doesn't
        # conflict with the transport's own "keep only the newest" discipline.
        self.history = ObservationHistory(length=self.policy.obs_horizon)

        self._stop = threading.Event()
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)

    def _recv_loop(self) -> None:
        while not self._stop.is_set():
            obs = self.obs_sub.recv(timeout_ms=50)
            if obs is not None:
                self.buffer.put(obs)

    def run(self) -> None:
        self._recv_thread.start()
        logger.info("policy server started (backend=%s), waiting for first observation", type(self.policy).__name__)
        last_seen_frame_id: int | None = None
        warmed_up = False
        try:
            while not self._stop.is_set():
                obs = self.buffer.wait_for_new(last_seen_frame_id, timeout=1.0)
                if obs is None:
                    continue
                last_seen_frame_id = obs["frame_id"]
                self.history.push(obs)
                if not self.history.is_ready():
                    continue  # still filling the observation-history window

                start = time.perf_counter()
                actions = self.policy.predict(self.history.stacked())
                elapsed_ms = (time.perf_counter() - start) * 1_000

                if not warmed_up:
                    logger.info("warm-up inference: %.2f ms", elapsed_ms)
                    warmed_up = True
                else:
                    logger.debug("steady-state inference: %.2f ms", elapsed_ms)

                self.action_pub.send(
                    {
                        "timestamp_us": now_us(),
                        "frame_id": obs["frame_id"],
                        "actions": actions,
                        "horizon": self.policy.horizon,
                        "action_dim": self.policy.action_dim,
                    }
                )
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        self._stop.set()
        if self._recv_thread.is_alive():
            self._recv_thread.join(timeout=2.0)
        self.obs_sub.close()
        self.action_pub.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the policy server.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    config = yaml.safe_load(args.config.read_text())
    PolicyServer(config).run()


if __name__ == "__main__":
    main()
