#!/usr/bin/env python
"""Thin CLI entry point for the dashboard bridge server -- see dashboard/server.py.

The frontend (frontend/, a Vite dev server) is started separately; this only
runs the Python side that speaks ZeroMQ to the sim and WebSocket/REST to the
browser.
"""

import argparse
import os
import sys
from pathlib import Path

# See scripts/run_sim.py's comment on why this is needed for a direct
# `python scripts/run_dashboard.py` invocation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the dashboard bridge server.")
    parser.add_argument("--config", type=Path, default=Path("sim/env_config.yaml"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    os.environ["SIM_CONFIG_PATH"] = str(args.config)
    uvicorn.run("dashboard.server:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
