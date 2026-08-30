# Getting Started

## Build

Needs CMake 3.21+ and a C++23 compiler. CI and the container use LLVM/Clang 23;
GCC 13+ is also supported.

```bash
git clone <repo> && cd FlowEdge
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Get a model

Download one to try, or convert your own (see [Converter](guides/converter)).

```bash
mkdir -p models
wget -qO models/mamba_flow.safetensors \
  "https://huggingface.co/ReForceMind/mamba_flow/resolve/main/mamba_flow.safetensors"
```

## Sample an action

```bash
./build/flow_sample models/mamba_flow.safetensors euler 10
# action_dim=8  solver=euler  NFE=10  action[0..2]=0.021476, 0.176867, 0.692564
```

Solver is `euler`, `heun`, or `rk4`. The last argument is the number of steps.

State can migrate between compatible streaming engines without exposing raw internal arrays:

```bash
./build/streaming_snapshot models/mamba_flow.safetensors
# snapshot_bytes=... migrated_exact=true output0=...
```

## Relay end-to-end demo

Build the optional local systems layer, then run its daemon/client/deadline/trace lifecycle:

```bash
FLOWEDGE_BUILD_DIR="$PWD/build-relay" ./scripts/build.sh Release -DFLOWEDGE_RELAY=ON
FLOWEDGE_BUILD_DIR="$PWD/build-relay" ./scripts/relay_demo.sh models/mamba_flow.safetensors
```

The demo starts two preallocated workers, submits through `RelayClient`, demonstrates a typed
deadline rejection, shuts the daemon down, inspects the portable trace, and replays completed actions.
Use `scripts/verify_all.sh` for the complete tests/examples/benchmarks/install-consumer gate.

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
