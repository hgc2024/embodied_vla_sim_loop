"""Stacks the last N observations into a policy's input, instead of
conditioning on only the single newest frame.

Inspired by diffusion_policy's n_obs_steps convention
(.reference/diffusion_policy-main/diffusion_policy/policy/base_image_policy.py,
not committed -- see README's "License and external assets"; not copied from,
just the same well-established idea: real manipulation policies often
condition on a short history so they can pick up on motion cues -- e.g. "is
the gripper still closing" -- that a single instant alone doesn't carry).
This project's Observation already reports joint_velocities directly (a real
robot's encoders give you that for free), so history stacking matters most
for images and general temporal context, not for recovering velocity we
already have explicitly.

This is a separate concept from the transport layer's "keep only the newest
frame" conflation (see zmq_publisher.py) -- that discards backlog so a slow
policy is never handed stale queued frames; this deliberately *keeps* the
last few frames the policy actually did see, once it's ready for each one.
They operate at different layers and don't conflict: the buffer below only
ever receives frames that already made it past the transport's conflation.
"""

from __future__ import annotations

import collections
from typing import Any

import numpy as np

from policy_server.normalizer import IdentityNormalizer, NormalizerLike


class ObservationHistory:
    """Fixed-length ring buffer of the last `length` observations.

    `stacked()` returns arrays with a new leading time axis, `(length, ...)`
    -- matching the `(B, To, *)` convention real image/diffusion policies
    expect for observation input (see base_image_policy.py's docstring),
    just without the batch axis, since this project runs one environment at
    a time. Non-array fields (task_instruction, frame_id, timestamp_us)
    aren't stacked -- there's exactly one current value of those, not a
    history of them worth keeping.
    """

    STACKED_KEYS = ("rgb", "depth", "joint_positions", "joint_velocities")
    PASSTHROUGH_KEYS = ("task_instruction", "frame_id", "timestamp_us")

    def __init__(self, length: int, state_normalizer: NormalizerLike | None = None) -> None:
        if length < 1:
            raise ValueError(f"history length must be >= 1, got {length}")
        self.length = length
        self.state_normalizer = state_normalizer or IdentityNormalizer()
        self._buffer: collections.deque[dict[str, Any]] = collections.deque(maxlen=length)

    def reset(self) -> None:
        self._buffer.clear()

    def push(self, obs: dict[str, Any]) -> None:
        self._buffer.append(obs)

    def is_ready(self) -> bool:
        """False until `length` real observations have been pushed. There is
        no principled way to pad *before* the first real observation, so
        callers should wait rather than get a padded-with-nothing history."""
        return len(self._buffer) == self.length

    def stacked(self) -> dict[str, Any]:
        if not self._buffer:
            raise RuntimeError("no observations pushed yet")
        frames = list(self._buffer)
        # Pad by repeating the oldest available frame -- the standard
        # convention for "there is no before the first frame" (matches how a
        # real rollout's first few steps are handled elsewhere in this space).
        while len(frames) < self.length:
            frames.insert(0, frames[0])

        result: dict[str, Any] = {
            key: np.stack([f[key] for f in frames], axis=0)
            for key in self.STACKED_KEYS
            if key in frames[0]
        }
        if "joint_positions" in result:
            result["joint_positions"] = self.state_normalizer.normalize(result["joint_positions"])
        for key in self.PASSTHROUGH_KEYS:
            if key in frames[-1]:
                result[key] = frames[-1][key]
        return result
