<div align="center">
<img src="assets/surfingtux.png" alt="FlowEdge" width="820">
<h1>FlowEdge</h1>
<p><b>Real-time flow-matching action-head inference for robotics</b></p>
<p>
  ⚙️&nbsp;<b>CPU</b>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <img alt="CUDA" height="22" src="assets/nvidia.svg">&nbsp;<b>CUDA</b>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <img alt="Tenstorrent" height="22" src="assets/tenstorrent.jpg">&nbsp;<b>Tenstorrent</b>
</p>
</div>

***

FlowEdge is a custom C++ inference engine designed to execute flow-matching action heads for robotics:
- zero external dependencies
- decoupled compute backends
- native `.safetensors` loading, plus a torch/HF checkpoint converter
- C API and Python bindings
- verification against PyTorch

It’s inspired by `ggml` (minimalism and performance) and `PyTorch` (abstractions), but stays focused on edge robotics with hard real-time latency constraints and zero dynamic allocations.

## How FlowEdge compares

- **PyTorch:** PyTorch is designed for training and general-purpose inference. FlowEdge is significantly faster for small control models due to zero interpreter overhead and static memory graphs (see the Performance section).
- **ONNX Runtime:** ONNX is a massive framework with heavy dependencies. FlowEdge compiles to a tiny static binary and executes with zero heap allocations on the hot path.
- **ggml / llama.cpp:** While `ggml` is optimized for LLM text generation, FlowEdge is built for robotics: prioritizing low-latency continuous control (flow-matching ODE solvers) over auto-regressive token generation.

## Accelerators

FlowEdge supports the following hardware accelerators:
- ☑ CPU (AVX2 / NEON)
- ☐ CUDA
- ☐ Metal
- ☐ Vulkan

## Architectures & Heads

**Backbones:**
- ☑ Mamba selective-SSM
- ☐ Transformer

**Heads (Action Policies):**
- ☑ Flow-Matching CNF
- ☐ Diffusion Policy
- ☐ DiT

**ODE Solvers:**
- ☑ Euler
- ☑ Heun
- ☑ RK4

**Precision:**
- ☑ FP32
- ☑ BF16
- ☐ INT8

## Performance

**Mamba (Forward Pass)**

| Benchmark | Backend | PyTorch | FlowEdge | Speedup (vs PT) |
|-----------|---------|---------|----------|-----------------|
| BM_engine_forward | CPU | 73.0 ms | 28.0 ms | ~2.6x |
| BM_engine_forward | CUDA | TBD | TBD | TBD |
| BM_engine_forward | Metal | TBD | TBD | TBD |
| BM_engine_forward | Vulkan | TBD | TBD | TBD |
| BM_engine_forward | Tenstorrent | TBD | TBD | TBD |

*Mamba-130M (24 layers, d_model=768), seq_len=4, FP32, single-threaded CPU. Regenerate with `./scripts/bench.sh`; numbers depend on model and host.*

**Current CPU Runtime Smoke**

| Benchmark | Model | PyTorch | FlowEdge | Notes |
|-----------|-------|---------|----------|-------|
| Flow action latency, mean | `mamba_flow`, 2 layers, d_model=128 | 1613 us | 368 us | Euler, NFE=10 |
| Flow action latency, p99 | `mamba_flow`, 2 layers, d_model=128 | 2978 us | 680 us | 0 hot-loop allocations |
| C-ABI engine forward | `mamba_flow`, seq_len=4 | 2.134 ms | 0.085 ms | CPU smoke bench |


## Python Installation

You can install the Python bindings directly via pip (requires CMake 3.21+ and a C++23 compiler):

```bash
pip install .
```

Then in Python:

```python
import numpy as np
import flowedge

e = flowedge.Engine("models/mamba_flow.safetensors")
action = e.sample(prefix=[1, 2, 3, 4],
                  noise=np.random.randn(e.action_dim).astype("float32"),
                  steps=10, method="euler")
```

A Python script is also provided in [`examples/flow_sample.py`](examples/flow_sample.py).

## C++ Quick Start

**1. Get the code and build.**

Requires CMake 3.21+ and a C++23 compiler (GCC 13+ or Clang 16+).

```bash
git clone <repo> && cd FlowEdge
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

**2. Bring a model.** FlowEdge runs a Mamba flow-matching policy in `.safetensors` format
(tensors named `backbone.*` for the SSM and `flow.*` for the action head). You can download a pre-trained model to try the engine:

```bash
mkdir -p models
wget -qO models/mamba_flow.safetensors "https://huggingface.co/ReForceMind/mamba_flow/resolve/main/mamba_flow.safetensors"
```

Or bring your own - convert a torch / HuggingFace checkpoint into the FlowEdge layout:

```bash
python convert/convert.py path/to/checkpoint.safetensors models/mamba_flow.safetensors
```

See the [converter guide](docs/guides/converter.md) for supported architectures and how to add one.

**3. Sample an action chunk** (`euler|heun|rk4`, last argument = number of solver steps):

```bash
./build/flow_sample models/mamba_flow.safetensors euler 10
# action_dim=8  solver=euler  NFE=10  action[0..2]=0.021476, 0.176867, 0.692564
```

Also in `examples/`: `mamba_forward` (backbone hidden states / streaming).

## C API & CMake Integration

The C API is defined in [`src/api/engine.h`](src/api/engine.h).

If you install FlowEdge system-wide or use it via `FetchContent`, you can easily link it in your own `CMakeLists.txt`:

```cmake
find_package(FlowEdge REQUIRED)
target_link_libraries(my_robot_node PRIVATE FlowEdge::flowedge_engine)
```

```c
#include <api/engine.h>
#include <stdio.h>

fe_engine* e = fe_engine_load("your_model.safetensors");
if (!e) {
    printf("Error: %s\n", fe_engine_last_error());
    return 1;
}

size_t n = fe_engine_action_dim(e);
float noise[n], action[n];                        // noise ~ N(0,1)
int rc = fe_engine_sample(e, (int32_t[]){1, 2, 3, 4}, 4, noise, /*steps=*/10, /*0=euler*/0, action);

if (rc != 0) {
    printf("Engine error: %s\n", fe_engine_last_error());
}

fe_engine_free(e);
```
