"""Tests for the policy-server-side scaffolding: normalization and
observation-history stacking (see policy_server/normalizer.py and
policy_server/observation_history.py for the design rationale)."""

from __future__ import annotations

import numpy as np
import pytest

from policy_server.normalizer import IdentityNormalizer, Normalizer
from policy_server.observation_history import ObservationHistory
from policy_server.ring_buffer import LatestObservationBuffer
from policy_server.vla_wrapper import DummyPolicy


def test_normalizer_roundtrip():
    data = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
    normalizer = Normalizer.fit(data)

    normalized = normalizer.normalize(data)
    np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(normalizer.denormalize(normalized), data, atol=1e-5)


def test_normalizer_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        Normalizer(mean=np.zeros(3), std=np.zeros(2))


def test_identity_normalizer_is_a_noop():
    x = np.array([1.0, -2.0, 3.5])
    identity = IdentityNormalizer()
    np.testing.assert_array_equal(identity.normalize(x), x)
    np.testing.assert_array_equal(identity.denormalize(x), x)


def _make_obs(frame_id: int) -> dict:
    return {
        "rgb": np.full((2, 2, 3), frame_id, dtype=np.uint8),
        "joint_positions": np.array([float(frame_id)], dtype=np.float32),
        "joint_velocities": np.array([0.0], dtype=np.float32),
        "task_instruction": "pick up the block",
        "frame_id": frame_id,
        "timestamp_us": frame_id * 1000,
    }


def test_observation_history_not_ready_until_full():
    history = ObservationHistory(length=3)
    assert not history.is_ready()
    history.push(_make_obs(1))
    assert not history.is_ready()
    history.push(_make_obs(2))
    history.push(_make_obs(3))
    assert history.is_ready()


def test_observation_history_pads_by_repeating_oldest():
    """Stacking before the buffer is full still works (used internally to
    build the shape), padding with the oldest frame rather than crashing."""
    history = ObservationHistory(length=3)
    history.push(_make_obs(5))
    stacked = history.stacked()
    np.testing.assert_array_equal(stacked["joint_positions"], [[5.0], [5.0], [5.0]])


def test_observation_history_stacks_in_order():
    history = ObservationHistory(length=3)
    for frame_id in (1, 2, 3):
        history.push(_make_obs(frame_id))
    stacked = history.stacked()

    np.testing.assert_array_equal(stacked["joint_positions"], [[1.0], [2.0], [3.0]])
    assert stacked["rgb"].shape == (3, 2, 2, 3)
    # Non-array fields: latest value only, not stacked.
    assert stacked["frame_id"] == 3
    assert stacked["task_instruction"] == "pick up the block"


def test_observation_history_applies_state_normalizer():
    normalizer = Normalizer(mean=np.array([2.0]), std=np.array([1.0]))
    history = ObservationHistory(length=1, state_normalizer=normalizer)
    history.push(_make_obs(5))
    stacked = history.stacked()
    np.testing.assert_allclose(stacked["joint_positions"], [[3.0]])  # (5 - 2) / 1


def test_observation_history_reset_clears_buffer():
    history = ObservationHistory(length=2)
    history.push(_make_obs(1))
    history.push(_make_obs(2))
    assert history.is_ready()
    history.reset()
    assert not history.is_ready()


def test_dummy_policy_declares_obs_horizon_one():
    policy = DummyPolicy(action_dim=4, horizon=8)
    assert policy.obs_horizon == 1

    history = ObservationHistory(length=policy.obs_horizon)
    history.push(_make_obs(1))
    assert history.is_ready()
    actions = policy.predict(history.stacked())
    assert actions.shape == (8, 4)


def test_latest_observation_buffer_still_works_with_history_wiring():
    """Sanity check that PolicyServer's two buffers (LatestObservationBuffer
    for transport, ObservationHistory for policy context) compose cleanly."""
    buffer = LatestObservationBuffer()
    history = ObservationHistory(length=2)

    for frame_id in (1, 2, 3):
        buffer.put(_make_obs(frame_id))
        history.push(buffer.latest())

    assert history.is_ready()
    np.testing.assert_array_equal(history.stacked()["joint_positions"], [[2.0], [3.0]])
