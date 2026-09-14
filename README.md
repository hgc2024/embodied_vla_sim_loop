# embodied-vla-sim-loop

An asynchronous simulation-to-policy control loop for embodied AI experiments.
The project is designed to connect an NVIDIA Isaac Lab or MuJoCo manipulation
environment to an open-source Vision-Language-Action (VLA) policy without making
the physics loop wait for model inference.

> [!IMPORTANT]
> The MuJoCo path (Option A) is implemented and runnable end to end:
> `pyproject.toml`, `proto/schema.proto`, `sim/`, `policy_server/`,
> `scripts/run_sim.py` / `scripts/run_policy.py`, and a live browser dashboard
> (`dashboard/` + `frontend/`, see "Dashboard" below) all exist, and the
> dashboard's default policy (`scripted_reach`) drives the arm toward the
> target object with a real (hand-written, not trained) controller. The
> MuJoCo environment uses a placeholder 4-DOF arm
> (`sim/assets/placeholder_arm.xml`), not a real Panda -- see that file's
> docstring. `policy_server/adapters/openpi_policy.py` is a real adapter
> against a live OpenPI checkpoint server (see "Choose a policy backend"),
> but this project ships neither a checkpoint nor OpenPI's model code.
> Isaac Lab support (`sim/isaac_env.py`), the native C++ ZeroMQ client
> (`src/ipc/`), the Docker image, CI, and the LeRobot adapter are still
> scaffold only; commands referencing those describe the intended workflow
> once they land.

## In plain terms

If "world models," "VLA policies," and "ZeroMQ" aren't your everyday
vocabulary, start here before the more technical sections below.

**What this actually is:** a simulated robot arm in a physics engine (MuJoCo),
plus a second program that decides how to move it (the "policy"), running as
two separate processes that talk to each other over messages instead of one
waiting on the other. There's also a live dashboard (a webpage) you can open
in a browser to watch it happen in real time and to poke at it (pause it,
reset it, etc.).

**Why two separate programs instead of one?** The whole point of this project
is speed mismatch: physics needs to update dozens of times per second to look
and behave right, but a real AI model deciding "what should the arm do next"
can take much longer to think. If the simulation had to pause and wait every
time it asked the AI a question, it would stutter badly. So instead, the
simulation keeps running on its own clock, and whenever the AI's answer is
ready, the simulation picks it up -- and if the AI takes too long, the arm
just holds its position rather than freezing or crashing.

**What you'll see if you open the dashboard right now:** the arm reaching
toward the orange block on the table, using a hand-written "move toward it"
controller (`scripted_reach`, the default) -- genuinely goal-directed motion,
but not a trained AI. Two pieces are still placeholders, filled in with the
simplest possible thing so the surrounding machinery could be built and
tested first:

- **The robot arm** is a simple 4-jointed shape drawn from scratch
  (`sim/assets/placeholder_arm.xml`), not a real robot model like a Franka
  Panda.
- **The controller** deciding how to move it is currently
  `compute_scripted_reach_action` -- straight-line proportional feedback
  toward the block's true position, which the controller is allowed to see
  directly because it's part of the simulator itself; a real deployed policy
  would only ever see the camera images and joint state, never ground truth
  like that. The caption "pick up the block" you'll see is a fixed label --
  the controller only reaches toward the block, it doesn't grasp it (the
  gripper isn't articulated).

Once a real robot model and a real trained policy are plugged in later (see
"Choose a policy backend" and "Implementation milestones"), the exact same
dashboard and plumbing will show the arm attempting the task from vision
alone -- nothing about the dashboard or the transport layer needs to change
for that.

### Reading the dashboard

| Panel | What it means in plain language |
| --- | --- |
| Scene (overview camera) | A third-person view of the whole arm, table, and target block, from a fixed camera watching the scene -- not attached to the arm. This is the main "does this look like it's doing something" view. |
| Wrist camera (RGB / Depth) | What a camera mounted on the arm's own wrist sees, shown smaller below the scene view. Because it's mounted *on* the arm rather than watching it from across the room, it's often an extreme close-up of whatever surface is nearest -- that's normal, not a rendering bug. "Depth" is the same view, but colored by distance instead of by color: closer surfaces get one color, farther ones another. |
| Control rate | How many times per second the simulation is currently updating. Should sit near 60. |
| Action source | **Scripted** = the hand-written reach controller is driving the arm (today's default). **Predicted** = a policy server (dummy or a real one, e.g. OpenPI) is driving it instead. **Fallback** = no fresh instruction arrived in time, so the arm is just holding its last position rather than doing anything erratic. None of these are errors; fallback is a deliberate safety behavior. |
| Action age | How old the instruction currently driving the arm is, in milliseconds. |
| Fallback steps | A running count of how many times the simulation has had to fall back to "just hold position" since it started. |
| Domain randomization | Whether small random variations (lighting, camera position, object weight, friction) are being applied each time the episode resets -- see "Domain randomization" below for why that matters. |
| Controls | Reset starts a new episode. Pause freezes the physics entirely (the arm stops moving, camera stops updating). Step advances exactly one instant while paused, for inspecting things frame by frame. |
| Joint state | The arm's own sense of its joint angles and speeds -- the numbers a real robot's internal sensors would report. |

### A few terms used elsewhere in this document

- **Policy**: whatever is deciding what action to take next, given what the
  camera/sensors currently show. Can be as simple as random numbers, a
  hand-written controller (`scripted_reach`, today's default -- strictly
  speaking not a "policy" in the learned sense, but the same role in the
  architecture), or a full trained AI model.
- **VLA (Vision-Language-Action) policy**: an AI model that takes in a camera
  image and a text instruction (like "pick up the block") and outputs robot
  actions -- the eventual replacement for the dummy policy.
- **Action chunk**: instead of asking the policy for one action at a time
  (slow), it's asked for a short sequence of several future actions at once,
  which the simulator can keep executing even while the next request is
  still being worked on.
- **Domain randomization**: deliberately varying things like lighting, object
  weight, and friction slightly every episode, so that a policy trained in
  simulation doesn't just memorize one exact simulated scene and fail on
  anything slightly different (including, eventually, the real world).
- **ZeroMQ / message passing**: the library used for the two programs
  (simulation and policy) to send data back and forth quickly without one
  blocking on the other.

## Goals

- Step simulation continuously at 60 Hz or faster.
- Run policy inference asynchronously at approximately 20 Hz.
- Exchange observations and action chunks through non-blocking ZeroMQ sockets.
- Use Protocol Buffers as the language-neutral wire format.
- Predict short action horizons so the controller can continue while a newer
  observation is being processed.
- Support visual and physical domain randomization for sim-to-real experiments.
- Start with a dummy/scripted baseline, then plug in a real OpenPI (implemented)
  or LeRobot (not yet) checkpoint.

This is a research prototype, not a safety-rated robot controller. Test policies
in simulation before connecting real hardware, and add workspace limits, velocity
limits, collision handling, an emergency stop, and an independent watchdog before
any real-robot deployment.

## Architecture

```text
Simulation process (60+ Hz)                 Policy process (~20 Hz)

  Isaac Lab / MuJoCo                         PyTorch / VLA adapter
  +----------------------+                   +----------------------+
  | RGB-D + 7-DoF state  | -- Observation ->| latest-frame buffer  |
  | action-chunk player  |<- ActionChunk ----| async inference      |
  | domain randomizer    |   ZeroMQ + Proto  | chunk predictor      |
  +----------------------+                   +----------------------+
           |                                          |
           +-- physics never waits for inference -----+
```

The simulator publishes timestamped observations. The policy server keeps only
the freshest useful frames, performs inference, and returns a horizon of actions.
The simulator consumes that horizon one control step at a time and blends a new
chunk into the active one when it arrives. Stale messages are discarded by
`frame_id` and timestamp rather than queued indefinitely.

The initial wire schema is:

```protobuf
syntax = "proto3";
package embodied;

message Observation {
  uint64 timestamp_us = 1;
  uint64 frame_id = 2;
  bytes rgb_data = 3;
  bytes depth_data = 4;
  repeated float joint_positions = 5;
  repeated float joint_velocities = 6;
  string task_instruction = 7;
}

message ActionChunk {
  uint64 timestamp_us = 1;
  uint64 frame_id = 2;
  repeated float actions = 3;
  uint32 horizon = 4;
  uint32 action_dim = 5;
}
```

Shape, encoding, units, coordinate frames, and endianness must be fixed in the
implementation rather than inferred from byte lengths. The planned defaults are
RGB `uint8` HWC, depth `float32` in metres, joint positions in radians, joint
velocities in radians per second, and little-endian array storage.

## Planned repository layout

```text
embodied-vla-sim-loop/
├── .github/workflows/ci.yml
├── docker/Dockerfile.isaac
├── proto/schema.proto
├── sim/
│   ├── env_config.yaml
│   ├── isaac_env.py
│   ├── mujoco_env.py
│   ├── domain_randomizer.py
│   └── zmq_publisher.py
├── policy_server/
│   ├── server.py
│   ├── vla_wrapper.py
│   └── ring_buffer.py
├── dashboard/
│   └── server.py
├── frontend/            # React + Vite dashboard UI, see frontend/README.md
├── src/ipc/
│   ├── zmq_client.cpp
│   └── zmq_server.cpp
├── scripts/
│   ├── compile_proto.sh
│   ├── run_sim.py
│   ├── run_policy.py
│   └── run_dashboard.py
├── tests/
│   ├── test_ipc.py
│   └── test_policy_latency.py
├── CMakeLists.txt
├── pyproject.toml
└── README.md
```

## Host requirements

The recommended host is Ubuntu 22.04 or newer, either installed directly or used
through WSL2. Isaac Lab and full VLA checkpoints are GPU-heavy; MuJoCo plus the
dummy policy is the intended smoke-test path for machines without an NVIDIA GPU.

### Required external software

Install these before creating the Python environment:

| Dependency | Why it is needed | Installation / download |
| --- | --- | --- |
| Git | Clone this project and optional policy repositories | [git-scm.com/downloads](https://git-scm.com/downloads) |
| Python | Core runtime; use the version required by the chosen simulator | [python.org](https://www.python.org/downloads/) or Miniforge |
| Node.js (only for the dashboard) | Build/run the `frontend/` dashboard UI | [nvm](https://github.com/nvm-sh/nvm) (`nvm install --lts`) or [nodejs.org](https://nodejs.org/); on WSL2 install a native Linux Node via nvm rather than relying on a Windows-side install reachable through `/mnt/c` -- it's slow and its `#!/usr/bin/env node` shebang scripts (Vite included) won't resolve `node` on `PATH` |
| `uv` or Miniforge | Isolated Python environments | [uv installation](https://docs.astral.sh/uv/getting-started/installation/) / [Miniforge releases](https://github.com/conda-forge/miniforge/releases) |
| Protocol Buffer compiler (`protoc`) | Generate Python and C++ bindings from `proto/schema.proto` | [official installation guide](https://protobuf.dev/installation/) |
| CMake and a C++17 compiler | Build the optional native ZeroMQ clients | `build-essential cmake pkg-config` on Ubuntu; Visual Studio Build Tools on Windows |
| ZeroMQ development library | Link the optional C++ transport | `libzmq3-dev` and `cppzmq-dev` on Ubuntu; [ZeroMQ downloads](https://zeromq.org/download/) elsewhere |
| NVIDIA driver | Isaac Lab and CUDA policy inference | [NVIDIA driver downloads](https://www.nvidia.com/Download/index.aspx) |

For the container workflow, also install [Docker Engine](https://docs.docker.com/engine/install/)
and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The CUDA toolkit does not normally need to be installed separately when the
selected PyTorch or Isaac package already supplies its CUDA runtime. The host
NVIDIA driver is still required.

On Ubuntu, the non-Python build dependencies can be installed with:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config \
  protobuf-compiler libprotobuf-dev libzmq3-dev cppzmq-dev

protoc --version
cmake --version
```

On Windows, `protoc` is available through `winget install protobuf`. The planned
`compile_proto.sh` script requires WSL2 or Git Bash; native PowerShell users can
run the equivalent `protoc` command shown below. For local IPC, Linux/WSL2 should
use `ipc://` while native Windows configurations should prefer loopback TCP.

## Choose a simulator

Only one simulator backend is required.

### Option A: MuJoCo (fastest setup)

The official Python wheel includes the MuJoCo native library, so a separate
MuJoCo download or license key is not required:

```bash
python -m pip install mujoco
python -c "import mujoco; print(mujoco.__version__)"
```

See the [official MuJoCo Python documentation](https://mujoco.readthedocs.io/en/stable/python.html)
for viewer and headless-rendering details. A Panda MJCF model and any textures
used by this project must either be committed under an asset-compatible license
or downloaded by a future asset script with its source and license recorded.

### Option B: NVIDIA Isaac Lab

Isaac Lab is tightly coupled to particular Python, PyTorch, Isaac Sim, NVIDIA
driver, and operating-system versions. Follow the
[current official installation matrix](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
instead of mixing independently installed CUDA and PyTorch packages. Verify that
Isaac Sim launches before installing this project into the same environment.

A typical released-package flow is:

```bash
# Create the Python version required by the selected Isaac Lab release.
conda create -n embodied-vla python=<required-python-version>
conda activate embodied-vla
python -m pip install --upgrade pip

# Use the exact command and version from the official Isaac Lab release docs.
python -m pip install "isaaclab[isaacsim,all]==<compatible-version>" \
  --extra-index-url https://pypi.nvidia.com

isaacsim
```

The first launch may download extension caches and request acceptance of the
NVIDIA Omniverse EULA. Do not copy an unpinned Isaac Lab command from this README
without checking the release matrix; compatibility changes between releases.

If using the planned container image instead, the intended commands are:

```bash
docker build -f docker/Dockerfile.isaac -t embodied-vla-sim-loop:isaac .
docker run --rm --gpus all --ipc=host \
  -v "$PWD:/workspace/embodied-vla-sim-loop" \
  embodied-vla-sim-loop:isaac
```

`docker/Dockerfile.isaac` does not exist yet. It must pin a compatible Isaac
Lab/Isaac Sim base image and expose a non-interactive EULA acceptance mechanism
consistent with NVIDIA's container instructions.

## Project environment

Clone and install the core project:

```bash
git clone https://github.com/hgc2024/embodied_vla_sim_loop.git
cd embodied_vla_sim_loop

python -m venv .venv
source .venv/bin/activate               # Linux, macOS, or WSL2
# .venv\Scripts\Activate.ps1            # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install -e ".[mujoco,dev]" # lightweight path
# python -m pip install -e ".[isaac,dev]" # inside the Isaac environment
```

The intended core dependencies are PyTorch, NumPy, Gymnasium, pyzmq, protobuf,
and OpenCV. Simulator and policy integrations should be optional extras so that
installing the MuJoCo smoke test does not pull in Isaac Sim or a full VLA stack.

Confirm CUDA visibility when using a GPU:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Install PyTorch using the command generated by the
[official PyTorch selector](https://pytorch.org/get-started/locally/) if the
simulator environment does not already provide it. Avoid replacing the PyTorch
build bundled or pinned by Isaac Lab.

## Generate Protocol Buffer bindings

The planned helper generates both language targets:

```bash
bash scripts/compile_proto.sh
```

Its equivalent `protoc` invocation is:

```bash
mkdir -p generated/python generated/cpp
protoc -I proto \
  --python_out=generated/python \
  --cpp_out=generated/cpp \
  proto/schema.proto
```

Generated sources should be reproducible from the pinned compiler/runtime
versions. CI should regenerate them and fail on a diff, or generate them solely
during the build; the project should not silently mix an incompatible `protoc`
compiler and protobuf runtime.

## Choose a policy backend

Start with the dummy policy (small random actions) or `scripted_reach` (a
hand-written proportional controller that drives the end effector toward the
target object -- see `mujoco_env.py`'s `compute_scripted_reach_action`, the
dashboard's default). Both validate shapes, transport, scheduling, and
latency without downloading model weights or a trained model existing at
all; `scripted_reach` additionally exercises genuinely goal-directed motion.
Neither is a policy in the learned sense.

A real VLA model also needs a robot-specific observation transform, action
representation, normalization statistics, and checkpoint -- not just a model
file. Two pieces of scaffolding exist for this already, in `policy_server/`:

- `normalizer.py` -- per-dimension affine normalization (`Normalizer`) plus
  a no-op stand-in (`IdentityNormalizer`) with the same interface, so a real
  fitted normalizer drops in later without changing call sites.
- `observation_history.py` -- stacks the last `obs_horizon` observations a
  backend actually received into its `predict()` call, instead of only the
  single newest frame. Every backend implemented so far declares
  `obs_horizon = 1` (no history needed yet); a policy trained on multi-frame
  context would declare more.

Both are original code, not copied from anywhere, but the *shape* of the
idea in each is the same one used throughout this line of research: see
[Diffusion Policy](https://github.com/real-stanford/diffusion_policy)'s
`LinearNormalizer` and its `n_obs_steps` convention
(`diffusion_policy/policy/base_image_policy.py`) for the reference this was
checked against.

### OpenPI

`policy_server/adapters/openpi_policy.py` implements a real adapter against
[OpenPI](https://github.com/Physical-Intelligence/openpi)'s actual, published
`openpi-client` PyPI package -- not a reimplementation of their serving or
model code. It's genuine integration: install their library, point it at a
real running OpenPI policy server, and it works.

**Two things to know before using it:**

1. **Install `openpi-client` into its own environment, separate from this
   project's core `.venv`.** It pins `numpy<2.0.0`, which conflicts with the
   numpy 2.x this project's torch/mujoco need -- installing
   `.[openpi]` into the same venv as `.[mujoco]` will downgrade numpy and can
   break both. Run `policy_server` from that separate environment when using
   the `openpi` backend:

   ```bash
   python -m venv .venv-openpi
   source .venv-openpi/bin/activate
   python -m pip install -e ".[openpi]"
   ```

   (`typing_extensions` is pinned alongside `openpi-client` in that extra
   because `openpi-client` 0.1.2 imports it without declaring it as a
   dependency -- a real gap in their package metadata, found and worked
   around while verifying this adapter, not an assumption.)

2. **This project supplies neither a checkpoint nor OpenPI's model code.**
   You still need a real OpenPI policy server already running and reachable
   (OpenPI's own `scripts/serve_policy.py` from a real checkpoint, following
   OpenPI's own README) before pointing `policy.backend: openpi` at it. Set
   `policy.openpi.host`/`port` in `sim/env_config.yaml` to that server, and
   `image_key`/`state_key`/`prompt_key`/`action_key` to match whatever dict
   keys that specific checkpoint expects -- this project's Observation (one
   wrist camera, an N-joint arm, no gripper) doesn't match any published
   checkpoint's expected input exactly, so the defaults (matching OpenPI's
   `droid_policy.py`) are a starting point, not a guarantee.

**An architectural note worth being explicit about:** OpenPI's own reference
client (`openpi_client.action_chunk_broker.ActionChunkBroker`) calls `infer()`
synchronously and blocks the robot's control loop on the websocket
round-trip whenever the current action chunk runs out -- their real-time
story is "block once per chunk, amortized." This project's transport
(`sim/zmq_publisher.py`) is fully async instead, by design: physics never
blocks on `OpenPIPolicy.predict()`, however long it takes -- the sim just
keeps playing the last chunk, or falls back to holding position, until a
fresh one is ready (see the "Architecture" section above). Wrapping OpenPI's
synchronous client this way doesn't change how OpenPI itself works; it just
means the blocking happens on the policy server's own thread, never on the
simulator's.

Published OpenPI checkpoints can be large and some require substantial GPU
memory to serve; pin the OpenPI commit and checkpoint URI you're using. Do
not assume a checkpoint trained for ALOHA or DROID directly matches this
project's placeholder arm's action space just because both adapters exist --
the `image_key`/`state_key`/`action_key` mapping above only aligns the wire
format, not the actual robot geometry, joint count, or action convention.

### LeRobot

Follow [LeRobot's current installation guide](https://github.com/huggingface/lerobot/blob/main/docs/source/installation.mdx).
Install only the policy extra required by the selected checkpoint when possible.
For example, current LeRobot releases expose policy-specific extras such as
`smolvla` and `pi`; check upstream before choosing an extra because names and
Python requirements can change.

```bash
# Example only; run inside a LeRobot-compatible environment.
python -m pip install "lerobot[smolvla]"
python -m pip install "huggingface_hub[cli]"
hf auth login  # only when the selected Hub model is gated/private
```

LeRobot normally downloads Hub checkpoints into the Hugging Face cache on first
use. To keep large downloads off the system disk:

```bash
export HF_HOME=/path/to/large-disk/huggingface
```

Record the checkpoint repository, revision/commit, normalization statistics, and
expected camera/state/action keys in experiment configuration. Never benchmark a
floating `main` revision as though it were reproducible.


## Configuration

`sim/env_config.yaml` is the single source of truth for simulator, camera,
controller, transport, and randomization settings -- read that file directly
for the current, complete configuration (it's short and commented); the
highlights:

```yaml
simulator: mujoco                 # mujoco | isaac
physics_hz: 120
control_hz: 60

camera:
  width: 224
  height: 224
  depth_dtype: float32

policy:
  backend: scripted_reach         # scripted_reach | dummy | openpi | lerobot
  action_horizon: 16
  action_dim: 4                   # must match the loaded MJCF's actuator count
```

The real file additionally has an `ipc` block (five endpoints: observations,
actions, plus the dashboard-only status/commands/overview channels -- see
"Dashboard" below), a `policy.openpi` sub-block (only read when
`backend: openpi`, see "Choose a policy backend" above), and the
`domain_randomization` ranges (see that section below).

Use `tcp://127.0.0.1:5555`-style addresses instead of `ipc://` on native
Windows or when simulator and policy run in different containers. Bind only
to trusted interfaces; observations and actions are not authenticated or
encrypted by ZeroMQ by default.

## Run the loop

Use two terminals in the same activated environment.

Terminal 1 — start the policy server first:

```bash
python scripts/run_policy.py --config sim/env_config.yaml
```

Terminal 2 — start simulation:

```bash
python scripts/run_sim.py --config sim/env_config.yaml
```

The policy process should log model warm-up separately from steady-state latency.
The simulator should report physics/control frequency, current frame ID, action
age, dropped observations, and whether it is executing a predicted or safe fallback
action. Stopping or restarting the policy server must not block the physics loop.

## Dashboard

A live browser view of the running loop -- wrist camera (RGB + depth),
control rate, action age, predicted-vs-fallback mode, joint state, and
controls (reset episode, pause/resume/step, toggle domain randomization).
It's a third, non-blocking observer: `dashboard/server.py` subscribes to the
same Observation stream (plus a dashboard-only Status stream) the policy
server does, and can't slow down or interfere with the sim<->policy loop by
existing, misbehaving, or not running at all.

With the sim (and optionally the policy server) from the previous section
already running, in two more terminals:

```bash
# Terminal 3 -- bridge server (Python, talks ZeroMQ <-> WebSocket/REST)
python scripts/run_dashboard.py --config sim/env_config.yaml

# Terminal 4 -- frontend dev server (first time: cd frontend && npm install)
cd frontend && npm run dev
```

Open the URL Vite prints (`http://127.0.0.1:5173` by default). Reset/pause/
resume/step and the domain-randomization toggle send commands to the sim over
a dedicated channel (`ipc:///tmp/vla_commands.ipc` by default) that, unlike
observations and actions, is never dropped under backpressure -- see
`sim/zmq_publisher.py`'s `CommandPublisher`/`CommandSubscriber`.

## Domain randomization

Randomization is sampled at episode reset and should be seeded for reproducibility.
The initial ranges are:

| Property | Range |
| --- | --- |
| Camera translation | ±2 cm per configured axis |
| Camera rotation | ±3° per configured axis |
| Table/object appearance | Random licensed PBR textures and material parameters |
| Light direction/intensity | Task-configured distribution |
| Object mass | 0.85–1.15 × nominal mass |
| Contact friction | `mu` in `[0.2, 1.1]` |
| Joint damping | Task-configured variation around nominal values |

Log the random seed and sampled parameters with every episode. Validation should
reject non-physical values and prevent randomization from changing tensor shapes,
joint ordering, or controller units.

## Tests and benchmarks

The planned test suite is:

```bash
pytest -q
pytest -q tests/test_ipc.py
pytest -q tests/test_policy_latency.py -s
```

Latency instrumentation should use a monotonic clock within each host and report
distributions (median, p95, and p99), not only an average. GPU timing must
synchronize CUDA events before recording elapsed time. Measure at least:

1. Sensor capture to serialization.
2. Serialization plus IPC receipt.
3. Preprocessing and model forward pass.
4. Action deserialization to control application.
5. End-to-end observation-to-action round trip.

Target criteria:

| Metric | Target |
| --- | --- |
| Physics step rate | ≥60 Hz without waiting for policy inference |
| Policy inference | ≤30 ms for one 16-step action chunk after warm-up |
| IPC serialization/deserialization round trip | <1 ms on the same host |
| End-to-end round trip | <50 ms after warm-up |
| Randomized pick-and-place success | >80% over a fixed evaluation suite |

These are acceptance targets, not measured results. Hardware, resolution, model,
precision, transport, and batch size must accompany every reported number. A
large VLA may not meet the inference target without compilation, TensorRT, a
smaller checkpoint, or a higher-end GPU.

## Common problems

- **Isaac imports fail:** launch scripts with the Python environment documented
  for the installed Isaac Lab release; do not mix it with an arbitrary venv.
- **No camera output on a headless host:** configure EGL/headless rendering and
  confirm the simulator's own headless example works before debugging ZeroMQ.
- **`Address already in use`:** stop the process that owns the endpoint. Remove a
  stale IPC socket file only after confirming no publisher is running.
- **Subscriber receives nothing at startup:** PUB/SUB drops messages before the
  subscription is established. Use a readiness handshake or ROUTER/DEALER when
  startup delivery matters.
- **Latency grows over time:** bound socket high-water marks, drain stale frames,
  keep only the newest observation, and verify image encoding is not blocking the
  physics thread.
- **CUDA out of memory:** lower image resolution, use the checkpoint's supported
  reduced precision, shorten the action horizon, or run simulation and policy on
  separate GPUs.
- **Actions look plausible but the arm diverges:** verify joint order, units,
  control mode, coordinate frames, action normalization, and checkpoint-specific
  transforms before tuning the controller.

## Implementation milestones

- [x] Add `pyproject.toml` with core, simulator, policy, and development extras.
- [x] Add and compile `proto/schema.proto` for Python (C++ target still pending -- no consumer until `src/ipc/` exists).
- [x] Implement the Gymnasium MuJoCo smoke-test environment (placeholder arm; swap in a licensed Panda asset before real experiments).
- [ ] Implement the Isaac Lab Panda environment and domain randomization (blocked on Isaac Sim GPU/Vulkan support under WSL2; see memory).
- [x] Implement bounded, non-blocking ZeroMQ transport and stale-frame handling.
- [x] Implement the thread-safe latest-observation ring buffer.
- [x] Implement dummy, scripted-reach, and OpenPI policy adapters. LeRobot not started.
- [x] Implement action-chunk interpolation and safe fallback behavior.
- [x] Add a live browser dashboard (`dashboard/` + `frontend/`) with reset/pause/step/domain-randomization controls.
- [x] Add observation-history stacking and a normalization scaffold (`policy_server/observation_history.py`, `normalizer.py`).
- [x] Add IPC and policy-server unit tests (`tests/test_ipc.py`, `tests/test_policy_server.py`). Latency and soak tests not started.
- [ ] Pin and document a reproducible container image.
- [ ] Publish measured benchmarks and a randomized evaluation protocol.

## License and external assets

No project license has been added yet. Add one before distributing source or
artifacts. Robot meshes, textures, datasets, simulator assets, model code, and
model checkpoints retain their upstream licenses and may impose additional usage
conditions. Keep an asset manifest containing each source URL, exact revision,
checksum, license, and local destination; do not commit gated checkpoints or
credentials to Git.

`.reference/` (gitignored, never committed) holds full local checkouts of
[Diffusion Policy](https://github.com/real-stanford/diffusion_policy) (MIT)
and [OpenPI](https://github.com/Physical-Intelligence/openpi) (Apache-2.0),
kept for design-pattern analysis while building `policy_server/normalizer.py`,
`observation_history.py`, and `adapters/openpi_policy.py`. Both licenses would
permit direct reuse with attribution, but this project's own code in those
files is original, written against the public interfaces those reference
codebases expose (their real `openpi-client` PyPI package, and the general
shape of `LinearNormalizer`/`n_obs_steps`), not copied from their source.

