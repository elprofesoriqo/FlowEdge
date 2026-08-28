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

## Python

```bash
pip install .
```

```python
import numpy as np, flowedge
e = flowedge.Engine("models/mamba_flow.safetensors")
a = e.sample(prefix=[1, 2, 3, 4],
             noise=np.random.randn(e.action_dim).astype("float32"),
             steps=10, method="euler")
```
