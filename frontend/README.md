# Dashboard frontend

Live read-only + control dashboard for the sim loop: wrist camera (RGB +
depth), loop telemetry (control rate, action age, predicted-vs-fallback
mode), joint state, and controls (reset/pause/resume/step, domain
randomization toggle). See the root [README](../README.md#dashboard) for how
this fits into the rest of the project and how to run the full stack.

Talks to `dashboard/server.py` (a separate Python process) over WebSocket
(`/ws`, live frames + status) and REST (`/api/command/*`, control actions).
It has no direct connection to the sim or policy processes -- see that
file's docstring for why.

## Develop

```bash
npm install    # first time only
npm run dev    # http://127.0.0.1:5173
```

Requires the dashboard bridge server running (`python scripts/run_dashboard.py`)
and, for anything to actually appear, the sim (`python scripts/run_sim.py`).

`VITE_DASHBOARD_API_URL` / `VITE_DASHBOARD_WS_URL` env vars override the
default `http://127.0.0.1:8000` / `ws://127.0.0.1:8000/ws` if the bridge
server runs on a different host/port.

## Build

```bash
npm run build   # type-checks then builds to dist/
```

There is no deploy target yet -- this is a local development/debugging tool,
not a hosted app.
