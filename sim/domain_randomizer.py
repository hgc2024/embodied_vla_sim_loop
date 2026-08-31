"""Visual/physical domain randomization, sampled once per episode reset.

Perturbations are always applied relative to the model's *nominal* values
(captured at construction time), never to the previous episode's randomized
values -- otherwise repeated resets would drift the model away from anything
physically intended. Config values are validated eagerly so a non-physical
range (e.g. negative friction) fails at startup rather than silently
producing nonsense mid-run.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

_CAMERA_NAME = "wrist_cam"  # must match sim/mujoco_env.py's observation camera


class DomainRandomizer:
    def __init__(self, model: mujoco.MjModel, config: dict[str, Any]) -> None:
        self.model = model
        self.mass_scale = _require_range(config, "mass_scale", min_value=0.0)
        self.friction_range = _require_range(config, "friction", min_value=0.0)
        self.camera_translation_m = _require_nonnegative(config, "camera_translation_m")
        self.camera_rotation_deg = _require_nonnegative(config, "camera_rotation_deg")

        # Bodies/geoms at index 0 are the world; leave it untouched.
        self._nominal_body_mass = model.body_mass.copy()
        self._nominal_geom_friction = model.geom_friction.copy()

        self._cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, _CAMERA_NAME)
        if self._cam_id >= 0:
            self._nominal_cam_pos = model.cam_pos[self._cam_id].copy()
            self._nominal_cam_quat = model.cam_quat[self._cam_id].copy()

    def randomize(self, rng: np.random.Generator) -> None:
        m = self.model

        lo, hi = self.mass_scale
        scale = rng.uniform(lo, hi, size=m.nbody).astype(np.float32)
        m.body_mass[:] = self._nominal_body_mass
        m.body_mass[1:] *= scale[1:]  # skip world body (index 0)

        lo, hi = self.friction_range
        friction_scale = rng.uniform(lo, hi, size=m.ngeom).astype(np.float32)
        m.geom_friction[:] = self._nominal_geom_friction
        m.geom_friction[:, 0] *= friction_scale  # sliding friction only

        if self._cam_id >= 0:
            offset = rng.uniform(-1.0, 1.0, size=3) * self.camera_translation_m
            m.cam_pos[self._cam_id] = self._nominal_cam_pos + offset

            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis) + 1e-8
            angle = rng.uniform(-1.0, 1.0) * math.radians(self.camera_rotation_deg)
            jitter_quat = np.empty(4)
            mujoco.mju_axisAngle2Quat(jitter_quat, axis, angle)
            combined = np.empty(4)
            mujoco.mju_mulQuat(combined, self._nominal_cam_quat, jitter_quat)
            m.cam_quat[self._cam_id] = combined


def _require_range(
    config: dict[str, Any], key: str, min_value: float
) -> tuple[float, float]:
    value = config.get(key)
    if value is None or len(value) != 2:
        raise ValueError(f"domain_randomization.{key} must be a [low, high] pair")
    lo, hi = float(value[0]), float(value[1])
    if lo < min_value or hi < min_value or lo > hi:
        raise ValueError(
            f"domain_randomization.{key} = [{lo}, {hi}] is not a valid range "
            f"(expected {min_value} <= low <= high)"
        )
    return lo, hi


def _require_nonnegative(config: dict[str, Any], key: str) -> float:
    value = config.get(key)
    if value is None:
        raise ValueError(f"domain_randomization.{key} is required when enabled: true")
    value = float(value)
    if value < 0:
        raise ValueError(f"domain_randomization.{key} = {value} must be >= 0")
    return value
