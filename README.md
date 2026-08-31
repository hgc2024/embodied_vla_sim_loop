# embodied-vla-sim-loop

An asynchronous simulation-to-policy control loop for embodied AI experiments.
The project is designed to connect an NVIDIA Isaac Lab or MuJoCo manipulation
environment to an open-source Vision-Language-Action (VLA) policy without making
the physics loop wait for model inference.

> [!IMPORTANT]
> The MuJoCo path (Option A) is implemented and runnable end to end with the
> dummy policy: `pyproject.toml`, `proto/schema.proto`, `sim/`, `policy_server/`,
> and `scripts/run_sim.py` / `scripts/run_policy.py` all exist. The MuJoCo
> environment uses a placeholder 4-DOF arm (`sim/assets/placeholder_arm.xml`),
> not a real Panda -- see that file's docstring. Isaac Lab support
> (`sim/isaac_env.py`), the native C++ ZeroMQ client (`src/ipc/`), the Docker
> image, CI, and real policy backends (LeRobot/OpenPI) are still scaffold only;
> commands referencing those describe the intended workflow once they land.

## Goals

- Step simulation continuously at 60 Hz or faster.
- Run policy inference asynchronously at approximately 20 Hz.
- Exchange observations and action chunks through non-blocking ZeroMQ sockets.
- Use Protocol Buffers as the language-neutral wire format.
- Predict short action horizons so the controller can continue while a newer
  observation is being processed.
- Support visual and physical domain randomization for sim-to-real experiments.
- Start with a dummy PyTorch policy, then plug in LeRobot or OpenPI checkpoints.

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
├── src/ipc/
│   ├── zmq_client.cpp
│   └── zmq_server.cpp
├── scripts/
│   ├── compile_proto.sh
│   ├── run_sim.py
│   └── run_policy.py
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

Start with the dummy PyTorch policy. It validates shapes, transport, scheduling,
and latency without downloading model weights. A real VLA model also needs a
robot-specific observation transform, action representation, normalization
statistics, and checkpoint—not just a model file.

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

### OpenPI

OpenPI has its own dependency lock and setup workflow. Clone it outside this
repository and follow its [official README](https://github.com/Physical-Intelligence/openpi)
rather than merging its dependency set into the core environment:

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
uv sync
```

OpenPI can download published checkpoints on first use. These are large, and some
configurations require substantial GPU memory. Pin the OpenPI commit and checkpoint
URI. The clean integration boundary is a separately launched OpenPI inference
service with this project's adapter translating the observation/action schema.
Do not assume an OpenPI checkpoint trained for ALOHA or DROID directly matches a
Panda or Kinova action space.

## Configuration

`sim/env_config.yaml` is intended to be the single source of truth for simulator,
camera, controller, transport, and randomization settings. A representative
configuration is:

```yaml
simulator: mujoco                 # mujoco | isaac
robot: panda                      # panda | kinova_gen3
physics_hz: 120
control_hz: 60

camera:
  width: 224
  height: 224
  rgb_encoding: raw               # raw | jpeg | png
  depth_dtype: float32

ipc:
  observations: ipc:///tmp/vla_obs.ipc
  actions: ipc:///tmp/vla_actions.ipc
  high_water_mark: 2

policy:
  backend: dummy                  # dummy | lerobot | openpi
  checkpoint: null
  device: cuda
  precision: fp16
  action_horizon: 16
  compile: false

domain_randomization:
  enabled: true
  camera_translation_m: 0.02
  camera_rotation_deg: 3.0
  mass_scale: [0.85, 1.15]
  friction: [0.2, 1.1]
```

Use `tcp://127.0.0.1:5555` and `tcp://127.0.0.1:5556` instead of the two `ipc://`
addresses on native Windows or when simulator and policy run in different
containers. Bind only to trusted interfaces; observations and actions are not
authenticated or encrypted by ZeroMQ by default.

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
- [x] Implement dummy policy adapter. LeRobot and OpenPI adapters not started.
- [x] Implement action-chunk interpolation and safe fallback behavior.
- [x] Add IPC unit tests (`tests/test_ipc.py`). Latency and soak tests not started.
- [ ] Pin and document a reproducible container image.
- [ ] Publish measured benchmarks and a randomized evaluation protocol.

## License and external assets

No project license has been added yet. Add one before distributing source or
artifacts. Robot meshes, textures, datasets, simulator assets, model code, and
model checkpoints retain their upstream licenses and may impose additional usage
conditions. Keep an asset manifest containing each source URL, exact revision,
checksum, license, and local destination; do not commit gated checkpoints or
credentials to Git.

