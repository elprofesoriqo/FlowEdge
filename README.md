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
- zero runtime dependencies
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
- ☐ Tenstorrent / tt-metal

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

**Local smoke benchmark**

These numbers are from `models/mamba_flow.safetensors` on the current CPU path
with a 2-layer, `d_model=128`, `seq_len=4` model. Regenerate them on your host
with `./scripts/bench.sh models/mamba_flow.safetensors`; production numbers will
move with model shape, compiler, CPU topology, and memory bandwidth.

| Benchmark | PyTorch | FlowEdge | Speedup |
|-----------|---------|----------|---------|
| Flow action latency, mean | 1613 us | 368 us | 4.4x |
| Flow action latency, p99 | 2978 us | 680 us | 4.4x |
| C-ABI engine forward | 2.134 ms | 0.085 ms | 25.1x |

The hot loop reports zero allocations. Large projection matmuls can use the
arena-backed lock-free thread pool to increase DRAM bandwidth parallelism.

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
