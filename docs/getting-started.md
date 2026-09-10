# Getting started

## 1. Build

| Requirement | Version |
|---|---|
| CMake | 3.21+ |
| Compiler | C++23; Clang 23 or GCC 13+ |
| Optional | Python 3 + NumPy; Relay build for local IPC |

```bash
git clone https://github.com/elprofesoriqo/FlowEdge.git
cd FlowEdge
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

## 2. Download and run

```bash
mkdir -p models
wget -qO models/mamba_flow.safetensors \
  https://huggingface.co/ReForceMind/mamba_flow/resolve/main/mamba_flow.safetensors
./build/flow_sample models/mamba_flow.safetensors euler 10
```

| Input | Output |
|---|---|
| Token prefix | Deterministic action chunk |
| Solver | `euler`, `heun`, or `rk4` |
| Steps | Last argument; higher cost, usually better quality |

## Choose an integration

```{mermaid}
flowchart TD
  S[Start] --> Q{What do you already have?}
  Q -->|Tokens + full policy| C[flow_sample]
  Q -->|Condition vector| H[External/head-only API]
  Q -->|Python arrays| P[Python API]
  Q -->|Robot policy checkpoint| L[LeRobot adapter]
  Q -->|Multiple local clients| R[Relay]
  Q -->|Stateful work| J[Cooperative jobs]
```

| Need | Next page |
|---|---|
| Existing vision/VLA encoder | [External encoder](#external-encoder) |
| Python control loop | [Python](#python) and [Python API](api/python) |
| LeRobot SO-100/SO-101 seam | [LeRobot guide](guides/lerobot) |
| Local process service | [Relay quickstart](guides/relay-quickstart) |
| Worker migration | [Cooperative jobs](guides/cooperative-jobs) |

(external-encoder)=
## External encoder

```bash
./build/external_flow_sample models/mamba_flow.safetensors
```

| Caller owns | FlowEdge owns |
|---|---|
| Condition, noise, output buffers | Solver scratch and cancellation state |
| Observation encoding and normalization | Fixed model weights and dimensions |

The condition width and model identity must match the checkpoint. Use
[deployment profiles](architecture/deployment-profile) to make that contract explicit.

## Relay

```bash
FLOWEDGE_BUILD_DIR=build-relay ./scripts/build.sh Release -DFLOWEDGE_RELAY=ON
FLOWEDGE_BUILD_DIR=build-relay ./scripts/relay_demo.sh models/mamba_flow.safetensors
```

```{mermaid}
flowchart LR
  C[Client] -->|bounded IPC| D[Relay daemon]
  D --> E[EDF admission]
  E --> W[Preallocated workers]
  W --> I[Core inference]
  I --> T[Trace + metrics]
```

Use `scripts/verify_all.sh` for the complete tests, examples, benchmarks, and install-consumer gate.

## Cooperative jobs

```bash
./build-relay/cooperative_job_sample
./build-relay/routed_job_sample
./build-relay/mamba_relay_stream models/mamba_flow.safetensors
```

| Job kind | Contract |
|---|---|
| Iterative | Canonical state capsule at a work boundary |
| Streaming | Exact Mamba continuation and migration |
| Speculative | Cancellation and completion state |

(python)=
## Python

```bash
python -m pip install .
```

```python
import numpy as np
import flowedge

engine = flowedge.Engine("models/mamba_flow.safetensors")
prefix = np.array([1, 2, 3, 4], dtype=np.int32)
noise = np.zeros(engine.action_dim, dtype=np.float32)
action = np.empty(engine.action_dim, dtype=np.float32)
engine.sample_into(prefix, noise, action, steps=10, method="euler")
```

## Deployment rules

| Rule | Meaning |
|---|---|
| Allocation | Loading may allocate; supported inference/Relay hot paths allocate zero after setup |
| Ownership | One engine owns one mutable stream or active solve; use one per concurrent lane |
| Safety | Relay gates freshness, bounds, overlap, and deadlines; the robot still owns final safety |
| Scope | CPU is implemented; CUDA, Tenstorrent, Transformer, and external adapters remain planned |
