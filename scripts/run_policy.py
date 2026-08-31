#!/usr/bin/env python
"""Thin CLI entry point -- see policy_server/server.py for the implementation."""

import sys
from pathlib import Path

# See scripts/run_sim.py's comment: generated/ needs the repo root on
# sys.path when this file is run directly rather than via `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from policy_server.server import main

if __name__ == "__main__":
    main()
