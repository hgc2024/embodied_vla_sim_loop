"""Gymnasium MuJoCo smoke-test environment.

Wraps the placeholder arm in sim/assets/placeholder_arm.xml (see that file's
docstring -- it is not the Panda referenced elsewhere in the README/config;
swap in a licensed asset before running real experiments). This module only
deals in plain numpy/dict observations and actions; wire-format conversion
lives in zmq_publisher.py so this class has no protobuf/ZeroMQ dependency and
can be unit-tested or driven interactively on its own.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from sim.domain_randomizer import DomainRandomizer

DEFAULT_MODEL_PATH = Path(__file__).parent / "assets" / "placeholder_arm.xml"


class MujocoManipulationEnv(gym.Env):
    """Steps physics at `physics_hz`, exposes control at `control_hz`.

    One call to `step()` is one control decision: it holds the given action
    constant across `physics_hz / control_hz` physics substeps, matching the
    README's "physics never waits for inference" model where the simulator
    is the one deciding the pace, not the policy.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict[str, Any],
        model_path: str | Path = DEFAULT_MODEL_PATH,
        max_episode_steps: int = 500,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        physics_hz = config.get("physics_hz", 120)
        control_hz = config.get("control_hz", 60)
        if physics_hz % control_hz != 0:
            raise ValueError(
                f"physics_hz ({physics_hz}) must be an integer multiple of "
                f"control_hz ({control_hz}) so each control step holds an "
                f"action for a whole number of physics substeps."
            )
        self.physics_substeps = physics_hz // control_hz
        self.control_hz = control_hz
        self.max_episode_steps = max_episode_steps

        cam_cfg = config.get("camera", {})
        self.cam_width = cam_cfg.get("width", 224)
        self.cam_height = cam_cfg.get("height", 224)
        self.depth_dtype = np.dtype(cam_cfg.get("depth_dtype", "float32"))
        self._camera_name = "wrist_cam"

        self._rgb_renderer = mujoco.Renderer(
            self.model, height=self.cam_height, width=self.cam_width
        )
        self._depth_renderer = mujoco.Renderer(
            self.model, height=self.cam_height, width=self.cam_width
        )
        self._depth_renderer.enable_depth_rendering()

        # Controllable (hinge) joints, in qpos/qvel order -- excludes the
        # target object's freejoint, which is simulated but not observed
        # as part of the arm's proprioceptive state.
        self._qpos_idx: list[int] = []
        self._qvel_idx: list[int] = []
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                self._qpos_idx.append(self.model.jnt_qposadr[j])
                self._qvel_idx.append(self.model.jnt_dofadr[j])

        ctrl_range = self.model.actuator_ctrlrange.astype(np.float32)
        self.action_space = gym.spaces.Box(
            low=ctrl_range[:, 0], high=ctrl_range[:, 1], dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(
            {
                "rgb": gym.spaces.Box(
                    0, 255, (self.cam_height, self.cam_width, 3), dtype=np.uint8
                ),
                "depth": gym.spaces.Box(
                    0.0, np.inf, (self.cam_height, self.cam_width), dtype=np.float32
                ),
                "joint_positions": gym.spaces.Box(
                    -np.inf, np.inf, (len(self._qpos_idx),), dtype=np.float32
                ),
                "joint_velocities": gym.spaces.Box(
                    -np.inf, np.inf, (len(self._qvel_idx),), dtype=np.float32
                ),
            }
        )

        # Constructed whenever ranges are configured at all, independent of
        # `enabled` -- that flag only gates whether reset() applies it, so
        # the dashboard can toggle it on/off at runtime (set_domain_randomization
        # below) without needing the ranges re-supplied.
        dr_cfg = config.get("domain_randomization", {})
        self.domain_randomizer = DomainRandomizer(self.model, dr_cfg) if dr_cfg else None
        self.domain_randomization_enabled = bool(dr_cfg.get("enabled", False))

        self.task_instruction = "pick up the block"
        self._frame_id = 0
        self._episode_step = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        if self.domain_randomizer is not None and self.domain_randomization_enabled:
            self.domain_randomizer.randomize(self.np_random)
        mujoco.mj_forward(self.model, self.data)
        self._episode_step = 0
        return self._get_observation(), {}

    def set_domain_randomization(self, enabled: bool) -> None:
        if self.domain_randomizer is None:
            raise ValueError(
                "domain_randomization has no ranges configured (env_config.yaml's "
                "domain_randomization block is empty), so it can't be toggled on"
            )
        self.domain_randomization_enabled = enabled

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        )
        self.data.ctrl[:] = action
        for _ in range(self.physics_substeps):
            mujoco.mj_step(self.model, self.data)

        self._episode_step += 1
        truncated = self._episode_step >= self.max_episode_steps
        # No task-completion signal yet -- this is a plumbing smoke test,
        # not a scored task. A real reward/success condition belongs here
        # once a concrete manipulation task is defined.
        reward = 0.0
        terminated = False
        return self._get_observation(), reward, terminated, truncated, {}

    def _get_observation(self) -> dict[str, np.ndarray]:
        self._rgb_renderer.update_scene(self.data, camera=self._camera_name)
        rgb = self._rgb_renderer.render()

        self._depth_renderer.update_scene(self.data, camera=self._camera_name)
        depth = self._depth_renderer.render().astype(self.depth_dtype)

        self._frame_id += 1
        return {
            "rgb": rgb,
            "depth": depth,
            "joint_positions": self.data.qpos[self._qpos_idx].astype(np.float32),
            "joint_velocities": self.data.qvel[self._qvel_idx].astype(np.float32),
            "task_instruction": self.task_instruction,
            "timestamp_us": time.time_ns() // 1_000,
            "frame_id": self._frame_id,
        }

    def close(self) -> None:
        self._rgb_renderer.close()
        self._depth_renderer.close()
