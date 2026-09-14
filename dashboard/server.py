"""Bridge between the sim's ZeroMQ streams and a browser dashboard.

Subscribes to Observation + Status as a third, non-blocking client -- it
never touches the sim<->policy loop's sockets and can't slow that loop down
by existing or by being slow itself (CONFLATE means a stalled dashboard just
misses frames, nothing backs up). Commands flow the other way over a small
REST API, translated into the same JSON command messages scripts/run_sim.py
already knows how to drain.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
import zmq
import zmq.asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# See scripts/run_sim.py's comment: generated/ needs the repo root on
# sys.path when this file isn't reached through the installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.zmq_publisher import CommandPublisher, unpack_observation  # noqa: E402

logger = logging.getLogger("dashboard")

CONFIG_PATH = Path(os.environ.get("SIM_CONFIG_PATH", "sim/env_config.yaml"))
JPEG_QUALITY = 80
MAX_DEPTH_M = 3.0  # clip range for the depth colormap; not a real sensor limit


class ConnectionManager:
    """Fans one message out to every connected dashboard tab."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    async def broadcast(self, message: dict[str, Any]) -> None:
        if not self._clients:
            return
        data = json.dumps(message)
        dead = []
        for ws in self._clients:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)


manager = ConnectionManager()


def _encode_jpeg(frame_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf).decode("ascii")


def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    clipped = np.clip(depth, 0.0, MAX_DEPTH_M)
    normalized = (clipped / MAX_DEPTH_M * 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


async def _relay_observations(config: dict[str, Any]) -> None:
    ctx = zmq.asyncio.Context()
    socket = ctx.socket(zmq.SUB)
    socket.setsockopt(zmq.RCVHWM, 2)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.connect(config["ipc"]["observations"])

    cam_cfg = config["camera"]
    height, width = cam_cfg["height"], cam_cfg["width"]
    depth_dtype = np.dtype(cam_cfg["depth_dtype"])

    logger.info("relaying observations from %s", config["ipc"]["observations"])
    while True:
        data = await socket.recv()
        obs = unpack_observation(data, height=height, width=width, depth_dtype=depth_dtype)
        rgb_bgr = cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR)
        await manager.broadcast(
            {
                "type": "frame",
                "frame_id": obs["frame_id"],
                "timestamp_us": obs["timestamp_us"],
                "task_instruction": obs["task_instruction"],
                "joint_positions": obs["joint_positions"].tolist(),
                "joint_velocities": obs["joint_velocities"].tolist(),
                "rgb_jpeg": _encode_jpeg(rgb_bgr),
                "depth_jpeg": _encode_jpeg(_colorize_depth(obs["depth"])),
            }
        )


async def _relay_status(config: dict[str, Any]) -> None:
    ctx = zmq.asyncio.Context()
    socket = ctx.socket(zmq.SUB)
    socket.setsockopt(zmq.RCVHWM, 2)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.connect(config["ipc"]["status"])

    logger.info("relaying status from %s", config["ipc"]["status"])
    while True:
        data = await socket.recv()
        status = json.loads(data.decode("utf-8"))
        status["type"] = "status"
        await manager.broadcast(status)


_command_pub: CommandPublisher | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _command_pub
    config = yaml.safe_load(CONFIG_PATH.read_text())
    _command_pub = CommandPublisher(config["ipc"]["commands"])

    obs_task = asyncio.create_task(_relay_observations(config))
    status_task = asyncio.create_task(_relay_status(config))
    try:
        yield
    finally:
        obs_task.cancel()
        status_task.cancel()
        _command_pub.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        while True:
            # The dashboard doesn't send anything meaningful over this socket
            # (commands go through the REST API below) -- this just blocks
            # until the browser tab disconnects.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


def _send_command(command: dict[str, Any]) -> dict[str, str]:
    assert _command_pub is not None, "command publisher not initialized"
    _command_pub.send(command)
    return {"status": "sent"}


@app.post("/api/command/reset")
async def reset() -> dict[str, str]:
    return _send_command({"type": "reset"})


@app.post("/api/command/pause")
async def pause() -> dict[str, str]:
    return _send_command({"type": "pause"})


@app.post("/api/command/resume")
async def resume() -> dict[str, str]:
    return _send_command({"type": "resume"})


@app.post("/api/command/step")
async def step() -> dict[str, str]:
    return _send_command({"type": "step"})


class DomainRandomizationRequest(BaseModel):
    enabled: bool


@app.post("/api/command/domain_randomization")
async def set_domain_randomization(request: DomainRandomizationRequest) -> dict[str, str]:
    return _send_command({"type": "set_domain_randomization", "enabled": request.enabled})
