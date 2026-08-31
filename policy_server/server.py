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

from policy_server.ring_buffer import LatestObservationBuffer
from policy_server.vla_wrapper import DummyPolicy, PolicyBackend
from sim.zmq_publisher import ActionChunkPublisher, ObservationSubscriber, now_us

logger = logging.getLogger("policy_server")


def build_policy(config: dict[str, Any]) -> PolicyBackend:
    policy_cfg = config["policy"]
    backend = policy_cfg.get("backend", "dummy")
    if backend != "dummy":
        raise NotImplementedError(
            f"policy backend '{backend}' is not implemented yet; only 'dummy' is "
            "available until LeRobot/OpenPI adapters land (see README milestones)"
        )
    return DummyPolicy(
        action_dim=policy_cfg["action_dim"], horizon=policy_cfg["action_horizon"]
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

                start = time.perf_counter()
                actions = self.policy.predict(obs)
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
