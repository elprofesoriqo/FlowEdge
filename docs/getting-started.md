# Getting Started

Choose the shortest path for what you are building:

| Goal | Continue at |
|---|---|
| Run the included complete policy | [Sample an action](sample-an-action) |
| Supply conditions from your own encoder | [Use an external encoder](use-an-external-encoder) |
| Run a local inference service | [Relay end-to-end demo](relay-end-to-end-demo) |
| Adapt a stateful ML workload | [Generic cooperative jobs](generic-cooperative-jobs) |
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
```

## Get a model

Download one to try, or convert your own (see [Converter](guides/converter)).

```bash
mkdir -p models
wget -qO models/mamba_flow.safetensors \
  "https://huggingface.co/ReForceMind/mamba_flow/resolve/main/mamba_flow.safetensors"
```

(sample-an-action)=
## Sample an action

```bash
./build/flow_sample models/mamba_flow.safetensors euler 10
# action_dim=8  solver=euler  NFE=10  action[0..2]=0.021476, 0.176867, 0.692564
```

Solver is `euler`, `heun`, or `rk4`. The last argument is the number of steps.

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

(relay-end-to-end-demo)=
## Relay end-to-end demo

Build the optional local systems layer, then run its daemon/client/deadline/trace lifecycle:

```bash
FLOWEDGE_BUILD_DIR="$PWD/build-relay" ./scripts/build.sh Release -DFLOWEDGE_RELAY=ON
FLOWEDGE_BUILD_DIR="$PWD/build-relay" ./scripts/relay_demo.sh models/mamba_flow.safetensors
```

The demo starts two preallocated workers, submits through `RelayClient`, demonstrates a typed
deadline rejection, shuts the daemon down, inspects the portable trace, and replays completed actions.
Use `scripts/verify_all.sh` for the complete tests/examples/benchmarks/install-consumer gate.
See the [Relay quickstart](guides/relay-quickstart) for manual daemon/client commands, expected
outcomes, production settings, and troubleshooting.

(generic-cooperative-jobs)=
## Generic cooperative jobs

Relay also exposes a runtime-neutral cooperative contract for iterative, streaming, and speculative
workloads. The sample advances a toy iterative model, migrates its canonical state capsule to a fresh
instance, demonstrates streaming cancellation, and completes a speculative job:

```bash
./build-relay/cooperative_job_sample
./build-relay/routed_job_sample
./build-relay/mamba_relay_stream models/mamba_flow.safetensors
./build-relay/flowedge_cooperative_job_bench 100000
./build-relay/flowedge_mamba_stream_bench models/mamba_flow.safetensors 1000
./build-relay/flowedge_worker_drain_bench 10000
```

`routed_job_sample` drains a live lane and moves its job to a compatible worker.
`mamba_relay_stream` proves exact Mamba continuation on another engine. The benchmarks fail on a
hot-path allocation.
See [Cooperative jobs and state migration](guides/cooperative-jobs).

(python-quickstart)=
## Python

```bash
pip install .
```

```python
import numpy as np, flowedge
e = flowedge.Engine("models/mamba_flow.safetensors")
prefix = np.array([1, 2, 3, 4], dtype=np.int32)
a = e.sample(prefix=prefix,
             noise=np.random.randn(e.action_dim).astype("float32"),
             steps=10, method="euler")
```

## What to expect

- Model loading may allocate once for weights and runtime setup. The supported hot paths
  allocate no heap memory after engine/worker initialization.
- One engine owns one mutable stream or active solve; use separate engines for concurrent sessions.
- The implemented backend is CPU. Mamba + flow and converted LeRobot Diffusion Policy are available;
  CUDA, Tenstorrent, and Transformer work remain tracked.
- Relay shared-memory rings are local SPSC endpoints, not a distributed queue.
- Deadline admission is disabled until you provide a measured `--nfe-ns` value.
- FlowEdge improves predictability inside the runtime but does not replace OS-level real-time setup or
  the robot's final safety controller.
