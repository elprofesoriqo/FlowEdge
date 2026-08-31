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
- an optional deadline-aware Relay with shared-memory inference, generic cooperative jobs, and
  portable state migration
- verification against PyTorch

It’s inspired by `ggml` (minimalism and performance) and `PyTorch` (abstractions), but stays focused
on predictable edge-robotics latency and zero dynamic allocations on supported hot paths.

The allocation-free engine lives in `src/core/` and is exported to CMake consumers as
`FlowEdge::Core`. The optional Relay systems layer lives in `src/relay/`, depends on Core, and adds a
typed client, worker-aware EDF admission, portable traces and state capsules, generic iterative,
streaming, and speculative adapters, and preallocated parallel head workers. Core does not depend on
transport, telemetry, ROS, or daemon libraries.

## How FlowEdge compares

- **PyTorch:** PyTorch is designed for training and general-purpose inference. FlowEdge removes
  interpreter and hot-path allocator overhead for its supported fixed models; use the matched
  benchmark scripts before claiming a speedup.
- **ONNX Runtime:** ONNX is a massive framework with heavy dependencies. FlowEdge compiles to a tiny static binary and executes with zero heap allocations on the hot path.
- **ggml / llama.cpp:** While `ggml` is optimized for LLM text generation, FlowEdge is built for robotics: prioritizing low-latency continuous control (flow-matching ODE solvers) over auto-regressive token generation.

## Accelerators

FlowEdge supports the following hardware accelerators:
- ☑ CPU (AVX2 / NEON)
- ☐ CUDA
- ☐ Metal
- ☐ Vulkan

## Quick Start

Not sure which path fits? See the [capability guide](docs/capabilities.md) for complete-policy,
external-encoder, streaming, Relay, and custom cooperative-job workflows.

<details open>
<summary><b>C++: build and sample</b></summary>

Requires CMake 3.21+ and a C++23 compiler. Clang 23 and CMake 4.4 are validated on Windows; GCC 13 and CMake 3.28 are validated on Linux.

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

On Linux, `scripts/build.sh`, `scripts/test.sh`, `scripts/lint.sh`, and `scripts/bench.sh` run the same workflow. Set `FLOWEDGE_LATENCY_ITERS=5000` to shorten the latency sample during development.

Build Relay with `-DFLOWEDGE_RELAY=ON`; use `scripts/relay_bench.sh` for its allocation-checked
single-worker and multi-worker paths. The daemon accepts `--workers 1..8` independently of the Core
`--threads` setting; workers share one immutable checkpoint store. Begin with
`--workers 2 --threads 0 --placement compact` and measure compact versus spread on the deployment CPU.
`scripts/relay_demo.sh` exercises the typed client, deadline outcome, trace inspection, replay, and
Prometheus/JSON/OTLP metrics; `scripts/verify_all.sh` runs the complete local release gate.
`cooperative_job_sample` demonstrates runtime-neutral state migration and cancellation;
`routed_job_sample` demonstrates typed request/result messages and bounded adapter lookup; and
`flowedge_cooperative_job_bench` enforces zero allocations across both paths.

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

Matched local results on an Intel i7-9750H, Release, one CPU thread, measured 2026-08-31:

| Benchmark | Platform | FlowEdge | PyTorch | Speedup |
|---|---|---:|---:|---:|
| Backbone forward | Windows | 0.091 ms | 1.650 ms | 18.1x |
| Backbone forward | Linux | 0.071 ms | 0.990 ms | 13.9x |
| Euler action head, mean | Windows | 375.73 us | 1,384.08 us | 3.68x |
| Euler action head, mean | Linux | 395.18 us | 939.67 us | 2.38x |

See [performance](docs/performance.md) for exact commands and Relay results.
