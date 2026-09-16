# Getting Started

Choose the shortest path for what you are building:

| Goal | Continue at |
|---|---|
| Deploy a converted LeRobot Diffusion Policy | [LeRobot rollout](lerobot-rollout) |
| Deploy SmolVLA with a cached VLM | [SmolVLA cached expert](smolvla-cached-expert) |
| Supply conditions from your own encoder | [Use an external encoder](use-an-external-encoder) |
| Run the Mamba CI fixture | [Sample a Mamba action](sample-an-action) |
| Run a local inference service | [Relay end-to-end demo](relay-end-to-end-demo) |
| Use NumPy/Python | [Python](python-quickstart) |

See [Capabilities](capabilities) for the full implemented/planned matrix and current limitations.

## Build

Needs CMake 3.21+ and a C++23 compiler. CI and the container use LLVM/Clang 23;
GCC 13+ is also supported.

```bash
git clone https://github.com/elprofesoriqo/FlowEdge.git
cd FlowEdge
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
python -m pip install .
python -m pip install -e integrations/lerobot
```

(lerobot-rollout)=
## LeRobot rollout

This is the product path. Convert a pinned Diffusion Policy, then measure a
bounded control loop. The encoder, joint limits, and emergency-stop stay in
LeRobot or the robot adapter.

```bash
hf download lerobot/diffusion_pusht --revision 84a7c23178445c6bbf7e1a884ff497017910f653 \
  --local-dir models/diffusion_pusht
python convert/convert.py models/diffusion_pusht \
  models/diffusion_pusht.flowedge.safetensors --arch diffusion --dtype f32
flowedge-lerobot-rollout models/diffusion_pusht.flowedge.safetensors \
  --steps 10 --threads 4 --period-ms 10 --on-miss hold
```

`--on-miss` is `hold` (repeat the last successfully sent action; zeros if none),
`drop` (skip `send_action`), or `raise` (`DeadlineMissed`). See the
[LeRobot adapter](guides/lerobot) for `--policy.type=flowedge` and ARM/Jetson
commands.

The published CPU replay is still slower than PyTorch; treat the JSON
`p50_ms` / `missed_deadlines` as the number to beat, not a latency claim.

(smolvla-cached-expert)=
## SmolVLA cached expert

`--policy.type=flowedge_smolvla` runs the native Euler action expert. LeRobot
owns SigLIP + SmolLM and supplies the KV cache. This is not a native VLM.

```bash
python -m pip install -e 'integrations/lerobot[smolvla]'
python tools/verification/verify_smolvla_hybrid.py \
  models/smolvla_base/model.safetensors models/smolvla_base \
  bench/artifacts/smolvla/eslab-frame-000000.capture.npz \
  --module-path build
```

See the [Transformer / SmolVLA guide](guides/transformer-backbone).

(use-an-external-encoder)=
## Use an external encoder

If perception, a VLA, or another runtime already produces the model's condition vector, load a
head-only or compatible checkpoint and call the external-head path:

```bash
./build/external_flow_sample models/mamba_flow.safetensors
# condition_dim=128 action_dim=8 generation=7 resumable_exact=true
```

Your condition width and model digest must match the checkpoint. The caller owns condition, noise,
and output buffers; the engine owns solver scratch and cancellation state.

State can migrate between compatible streaming engines without exposing raw internal arrays:

```bash
./build/streaming_snapshot models/mamba_flow.safetensors
# snapshot_bytes=... migrated_exact=true output0=...
```

(sample-an-action)=
## Sample a Mamba action

Mamba + flow is the allocation/streaming CI fixture. It is not the LeRobot
product default.

```bash
mkdir -p models
wget -qO models/mamba_flow.safetensors \
  "https://huggingface.co/ReForceMind/mamba_flow/resolve/main/mamba_flow.safetensors"
./build/flow_sample models/mamba_flow.safetensors euler 10
# action_dim=8  solver=euler  NFE=10  action[0..2]=0.021476, 0.176867, 0.692564
```

Solver is `euler`, `heun`, or `rk4`. The last argument is the number of steps.

(relay-end-to-end-demo)=
## Relay end-to-end demo

Relay is optional. Build the local systems layer, then run its daemon/client/deadline/trace lifecycle:

```bash
FLOWEDGE_BUILD_DIR="$PWD/build-relay" ./scripts/build.sh Release -DFLOWEDGE_RELAY=ON
FLOWEDGE_BUILD_DIR="$PWD/build-relay" ./scripts/relay_demo.sh models/mamba_flow.safetensors
```

The demo starts two preallocated workers, submits through `RelayClient`, demonstrates a typed
deadline rejection, shuts the daemon down, inspects the portable trace, and replays completed actions.
Use `scripts/verify_all.sh` for the complete tests/examples/benchmarks/install-consumer gate.
See the [Relay quickstart](guides/relay-quickstart) for manual daemon/client commands.

(python-quickstart)=
## Python

```bash
pip install .
```

```python
import numpy as np, flowedge
e = flowedge.Engine("models/diffusion_pusht.flowedge.safetensors")
```

For the Mamba smoke checkpoint, `Engine.sample(prefix, noise, steps, method)`
accepts `euler`, `heun`, or `rk4`.

## What to expect

- Model loading may allocate once for weights and runtime setup. The supported hot paths
  allocate no heap memory after engine/worker initialization.
- One engine owns one mutable stream or active solve; use separate engines for concurrent sessions.
- The implemented backend is CPU. Converted LeRobot Diffusion Policy and the SmolVLA
  cached expert are the user paths; Mamba remains the CI fixture.
- Relay shared-memory rings are local SPSC endpoints, not a distributed queue.
- Deadline admission is disabled until you provide a measured `--nfe-ns` value.
- FlowEdge improves predictability inside the runtime but does not replace OS-level real-time setup or
  the robot's final safety controller.
