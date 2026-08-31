"""Thread-safe latest-observation buffer.

Holds at most one Observation: the newest one seen. A background thread
receiving off the ZeroMQ socket calls `put()`; the inference thread calls
`wait_for_new()` to block until something newer than what it already
processed shows up. This is where "discard by frame_id" actually happens on
the policy side (the transport layer's CONFLATE already discards backlog at
the socket level; this discards a same-or-older frame that raced in after a
newer one during the brief window between socket recv and put()).
"""

from __future__ import annotations

import threading
from typing import Any


class LatestObservationBuffer:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._obs: dict[str, Any] | None = None

    def put(self, obs: dict[str, Any]) -> None:
        with self._cond:
            if self._obs is not None and obs["frame_id"] <= self._obs["frame_id"]:
                return  # stale relative to what's already buffered
            self._obs = obs
            self._cond.notify_all()

    def latest(self) -> dict[str, Any] | None:
        with self._cond:
            return self._obs

    def wait_for_new(
        self, last_seen_frame_id: int | None, timeout: float | None = None
    ) -> dict[str, Any] | None:
        """Block until an observation newer than `last_seen_frame_id` arrives.

        Returns None on timeout, never on a stale/duplicate frame.
        """
        with self._cond:
            got = self._cond.wait_for(
                lambda: self._obs is not None and self._obs["frame_id"] != last_seen_frame_id,
                timeout=timeout,
            )
            return self._obs if got else None
