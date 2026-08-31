"""ZeroMQ transport: bounded, non-blocking, latest-frame-only.

Both directions (sim -> policy observations, policy -> sim action chunks)
use PUB/SUB with CONFLATE=1, which collapses the socket's send/receive
buffer to a single slot holding only the most recent message. That gives us
"keep only the newest observation" and "stale messages are discarded ...
rather than queued indefinitely" for free, instead of managing a bounded
queue by hand.

Known limitation (see README "Common problems"): a SUB socket that connects
after messages have already been published misses them -- there is no
delivery guarantee for the first message. This module does not add a
readiness handshake; it relies on the sim loop publishing continuously, so a
late subscriber simply picks up the next frame rather than needing the very
first one.

This module intentionally has no dependency on sim/mujoco_env.py: the pack/
unpack functions take and return plain dicts, so policy_server (a separate
process/package) can use the same wire format without importing simulator
code.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import zmq

from generated.python import schema_pb2


def pack_observation(obs: dict[str, Any]) -> bytes:
    msg = schema_pb2.Observation(
        timestamp_us=obs["timestamp_us"],
        frame_id=obs["frame_id"],
        rgb_data=np.asarray(obs["rgb"], dtype=np.uint8).tobytes(),
        depth_data=np.asarray(obs["depth"], dtype=np.float32).tobytes(),
        joint_positions=np.asarray(obs["joint_positions"], dtype=np.float32).tolist(),
        joint_velocities=np.asarray(obs["joint_velocities"], dtype=np.float32).tolist(),
        task_instruction=obs["task_instruction"],
    )
    return msg.SerializeToString()


def unpack_observation(
    data: bytes, *, height: int, width: int, depth_dtype: np.dtype = np.dtype(np.float32)
) -> dict[str, Any]:
    """Reconstruct arrays from raw bytes.

    Shape is not carried on the wire (see proto/schema.proto) -- the caller
    must supply the same height/width/depth_dtype the publisher was
    configured with (i.e. the shared env_config.yaml `camera` block).
    """
    msg = schema_pb2.Observation()
    msg.ParseFromString(data)
    rgb = np.frombuffer(msg.rgb_data, dtype=np.uint8).reshape(height, width, 3)
    depth = np.frombuffer(msg.depth_data, dtype=depth_dtype).reshape(height, width)
    return {
        "timestamp_us": msg.timestamp_us,
        "frame_id": msg.frame_id,
        "rgb": rgb,
        "depth": depth,
        "joint_positions": np.asarray(msg.joint_positions, dtype=np.float32),
        "joint_velocities": np.asarray(msg.joint_velocities, dtype=np.float32),
        "task_instruction": msg.task_instruction,
    }


def pack_action_chunk(chunk: dict[str, Any]) -> bytes:
    actions = np.asarray(chunk["actions"], dtype=np.float32).reshape(-1)
    msg = schema_pb2.ActionChunk(
        timestamp_us=chunk["timestamp_us"],
        frame_id=chunk["frame_id"],
        actions=actions.tolist(),
        horizon=chunk["horizon"],
        action_dim=chunk["action_dim"],
    )
    return msg.SerializeToString()


def unpack_action_chunk(data: bytes) -> dict[str, Any]:
    msg = schema_pb2.ActionChunk()
    msg.ParseFromString(data)
    actions = np.asarray(msg.actions, dtype=np.float32).reshape(msg.horizon, msg.action_dim)
    return {
        "timestamp_us": msg.timestamp_us,
        "frame_id": msg.frame_id,
        "actions": actions,
        "horizon": msg.horizon,
        "action_dim": msg.action_dim,
    }


class _ConflatingPublisher:
    def __init__(self, endpoint: str, high_water_mark: int = 2) -> None:
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, high_water_mark)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.bind(endpoint)

    def _send(self, data: bytes) -> None:
        self._socket.send(data, flags=zmq.NOBLOCK)

    def close(self) -> None:
        self._socket.close(linger=0)


class _ConflatingSubscriber:
    def __init__(self, endpoint: str, high_water_mark: int = 2) -> None:
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, high_water_mark)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(endpoint)

    def _recv(self, timeout_ms: int = 0) -> bytes | None:
        if self._socket.poll(timeout=timeout_ms) == 0:
            return None
        return self._socket.recv()

    def close(self) -> None:
        self._socket.close(linger=0)


class ObservationPublisher(_ConflatingPublisher):
    """Sim-side: publishes the latest Observation."""

    def send(self, obs: dict[str, Any]) -> None:
        self._send(pack_observation(obs))


class ObservationSubscriber(_ConflatingSubscriber):
    """Policy-side: receives the latest Observation."""

    def __init__(
        self,
        endpoint: str,
        *,
        height: int,
        width: int,
        depth_dtype: np.dtype = np.dtype(np.float32),
        high_water_mark: int = 2,
    ) -> None:
        super().__init__(endpoint, high_water_mark)
        self._height = height
        self._width = width
        self._depth_dtype = depth_dtype

    def recv(self, timeout_ms: int = 0) -> dict[str, Any] | None:
        data = self._recv(timeout_ms)
        if data is None:
            return None
        return unpack_observation(
            data, height=self._height, width=self._width, depth_dtype=self._depth_dtype
        )


class ActionChunkPublisher(_ConflatingPublisher):
    """Policy-side: publishes the latest predicted ActionChunk."""

    def send(self, chunk: dict[str, Any]) -> None:
        self._send(pack_action_chunk(chunk))


class ActionChunkSubscriber(_ConflatingSubscriber):
    """Sim-side: receives the latest ActionChunk."""

    def recv(self, timeout_ms: int = 0) -> dict[str, Any] | None:
        data = self._recv(timeout_ms)
        if data is None:
            return None
        return unpack_action_chunk(data)


def now_us() -> int:
    return time.time_ns() // 1_000
