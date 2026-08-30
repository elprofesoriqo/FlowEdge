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
- an optional deadline-aware shared-memory Relay for cross-process inference
- verification against PyTorch

It’s inspired by `ggml` (minimalism and performance) and `PyTorch` (abstractions), but stays focused on edge robotics with hard real-time latency constraints and zero dynamic allocations.

The allocation-free engine lives in `src/core/` and is exported to CMake consumers as
`FlowEdge::Core`. The optional Relay systems layer lives in `src/relay/`, depends on Core, and adds a
typed client, worker-aware EDF admission, portable traces, and preallocated parallel head workers. Core does not
depend on transport, telemetry, ROS, or daemon libraries.

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

## Quick Start

<details open>
<summary><b>C++: build and sample</b></summary>

Requires CMake 3.21+ and a C++23 compiler. Clang 23 and CMake 4.4 are validated on Windows; GCC 13 and CMake 3.28 are validated in WSL.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/flow_sample models/mamba_flow.safetensors euler 10
```

The checkpoint must contain `backbone.*` Mamba tensors and `flow.*` action-head tensors. Download the smoke-test model:

```bash
mkdir -p models
wget -qO models/mamba_flow.safetensors \
  https://huggingface.co/ReForceMind/mamba_flow/resolve/main/mamba_flow.safetensors
```
</details>

<details>
<summary><b>Python: install and sample</b></summary>

```bash
python -m pip install .
python examples/flow_sample.py models/mamba_flow.safetensors
```

The Python wheel builds the same C++ engine. Use `Engine.sample(prefix, noise, steps, method)` for `euler`, `heun`, or `rk4`.
</details>

<details>
<summary><b>Validate locally</b></summary>

```bash
cmake -S . -B build -DFLOWEDGE_TESTS=ON -DFLOWEDGE_BENCH=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
FLOWEDGE_MODEL=models/mamba_flow.safetensors ./build/flowedge_engine_bench
```

On WSL, `scripts/build.sh`, `scripts/test.sh`, `scripts/lint.sh`, and `scripts/bench.sh` run the same workflow. Set `FLOWEDGE_LATENCY_ITERS=5000` to shorten the latency sample during development.

Build Relay with `-DFLOWEDGE_RELAY=ON`; use `scripts/relay_bench.sh` for its allocation-checked
single-worker and multi-worker paths. The daemon accepts `--workers 1..8` independently of the Core
`--threads` setting; workers share one immutable checkpoint store. Begin with
`--workers 2 --threads 0 --placement compact` and measure compact versus spread on the deployment CPU.
`scripts/relay_demo.sh` exercises the typed client, deadline outcome, trace inspection, and replay;
`scripts/verify_all.sh` runs the complete local release gate.

Installed CMake consumers should link `FlowEdge::Core`; `FlowEdge::flowedge_engine` remains available as a compatibility target.
</details>

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

| Benchmark | Backend | PyTorch | FlowEdge | Speedup (vs PyTorch) |
|---|---|---:|---:|---:|
| `BM_engine_forward` | CPU | 0.983 ms | 0.120 ms | 8.19x |
| `BM_engine_forward` | CUDA | TBD | TBD | TBD |
| `BM_engine_forward` | Metal | TBD | TBD | TBD |
| `BM_engine_forward` | Vulkan | TBD | TBD | TBD |
| `BM_engine_forward` | Tenstorrent | TBD | TBD | TBD |
| Flow action latency, mean (Euler, NFE=10) | CPU | 962.39 us | 294.42 us | 3.27x |
| Flow action latency, p99 (Euler, NFE=10) | CPU | 1,472.40 us | 482.72 us | 3.05x |

| Measurements | FlowEdge | PyTorch |
|---|---|---|
| Engine forward | [`bench/engine_bench.cc`](bench/engine_bench.cc) | [`scripts/torch_ref.py`](scripts/torch_ref.py) (`bench`) |
| Flow action latency | [`bench/latency_bench.cc`](bench/latency_bench.cc) | [`scripts/torch_ref.py`](scripts/torch_ref.py) (`latency`) |

These numbers use the included two-layer smoke checkpoint on one host; use `scripts/bench.sh` and `scripts/ab_bench.sh` on the same idle machine for deployment decisions.
