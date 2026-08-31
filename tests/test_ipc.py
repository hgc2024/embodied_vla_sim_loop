"""Tests for the wire format (proto pack/unpack) and the ZeroMQ transport
(latest-frame-only pub/sub, per the README's "physics never waits for
inference" / "stale messages are discarded" requirements).
"""

from __future__ import annotations

import socket
import time

import numpy as np
import pytest

from sim.zmq_publisher import (
    ActionChunkPublisher,
    ActionChunkSubscriber,
    ObservationPublisher,
    ObservationSubscriber,
    pack_action_chunk,
    pack_observation,
    unpack_action_chunk,
    unpack_observation,
)


def make_observation(frame_id: int = 1, height: int = 4, width: int = 4) -> dict:
    return {
        "timestamp_us": 1_000_000,
        "frame_id": frame_id,
        "rgb": np.full((height, width, 3), 7, dtype=np.uint8),
        "depth": np.full((height, width), 1.5, dtype=np.float32),
        "joint_positions": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "joint_velocities": np.array([0.0, -0.1, 0.2], dtype=np.float32),
        "task_instruction": "pick up the block",
    }


def test_observation_roundtrip():
    obs = make_observation()
    restored = unpack_observation(pack_observation(obs), height=4, width=4)

    assert restored["frame_id"] == obs["frame_id"]
    assert restored["timestamp_us"] == obs["timestamp_us"]
    assert restored["task_instruction"] == obs["task_instruction"]
    np.testing.assert_array_equal(restored["rgb"], obs["rgb"])
    np.testing.assert_allclose(restored["depth"], obs["depth"])
    np.testing.assert_allclose(restored["joint_positions"], obs["joint_positions"])
    np.testing.assert_allclose(restored["joint_velocities"], obs["joint_velocities"])


def test_action_chunk_roundtrip():
    chunk = {
        "timestamp_us": 42,
        "frame_id": 7,
        "actions": np.arange(8, dtype=np.float32).reshape(4, 2),
        "horizon": 4,
        "action_dim": 2,
    }
    restored = unpack_action_chunk(pack_action_chunk(chunk))

    assert restored["frame_id"] == 7
    assert restored["horizon"] == 4
    assert restored["action_dim"] == 2
    np.testing.assert_allclose(restored["actions"], chunk["actions"])


@pytest.fixture
def endpoint() -> str:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"tcp://127.0.0.1:{port}"


def test_observation_pubsub_roundtrip(endpoint):
    pub = ObservationPublisher(endpoint, high_water_mark=2)
    sub = ObservationSubscriber(endpoint, height=4, width=4, high_water_mark=2)
    try:
        # Give the SUB socket time to connect and its subscription to
        # propagate before publishing -- see zmq_publisher.py's docstring on
        # this known PUB/SUB limitation (a message sent before that has no
        # delivery guarantee).
        time.sleep(0.2)
        sent = make_observation(frame_id=1)
        pub.send(sent)

        received = sub.recv(timeout_ms=2000)
        assert received is not None
        assert received["frame_id"] == 1
        np.testing.assert_array_equal(received["rgb"], sent["rgb"])
    finally:
        pub.close()
        sub.close()


def test_conflate_keeps_only_latest(endpoint):
    """Publishing faster than the subscriber reads must never build a backlog."""
    pub = ObservationPublisher(endpoint, high_water_mark=2)
    sub = ObservationSubscriber(endpoint, height=4, width=4, high_water_mark=2)
    try:
        time.sleep(0.2)
        for frame_id in range(1, 11):
            pub.send(make_observation(frame_id=frame_id))
        time.sleep(0.2)  # let every send land

        received = sub.recv(timeout_ms=2000)
        assert received is not None
        assert received["frame_id"] == 10, "subscriber should see only the newest frame"
        assert sub.recv(timeout_ms=100) is None, "nothing should be queued behind it"
    finally:
        pub.close()
        sub.close()


def test_action_chunk_pubsub_roundtrip(endpoint):
    pub = ActionChunkPublisher(endpoint, high_water_mark=2)
    sub = ActionChunkSubscriber(endpoint, high_water_mark=2)
    try:
        time.sleep(0.2)
        sent = {
            "timestamp_us": 99,
            "frame_id": 3,
            "actions": np.ones((2, 4), dtype=np.float32),
            "horizon": 2,
            "action_dim": 4,
        }
        pub.send(sent)

        received = sub.recv(timeout_ms=2000)
        assert received is not None
        assert received["frame_id"] == 3
        np.testing.assert_allclose(received["actions"], sent["actions"])
    finally:
        pub.close()
        sub.close()
