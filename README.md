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

Current local reference results on an Intel i7-9750H, Release builds, measured 2026-08-31:

| Benchmark | Windows Clang 23 | WSL 2 GCC 13 | Hot allocations |
|---|---:|---:|---:|
| Smoke-checkpoint backbone forward, median | 71 us | 71 us | fixed engine memory |
| Synthetic Euler action, 10 steps, p99 | 555.20 us | 542.60 us | 0 |
| Relay synchronous local path, p99 | 47.60 us | 40.10 us | 0 |
| Relay pool, 1 worker | 37,240 req/s | 55,719 req/s | 0 |
| Relay pool, 2 workers | 73,187 req/s | 109,507 req/s | 0 |
| Cooperative partial-run/migrate/finish | 179.23 ns/job | 306.52 ns/job | 0 |
| Generic registry route/run/result | 85.86 ns/job | 92.51 ns/job | 0 |

The pool benchmark maintains a queue and measures throughput under concurrency; the synchronous path
measures one request at a time. These are reference measurements, not portable deadline guarantees.
See the [full methodology, latency percentiles, environment, and reproduction commands](docs/performance.md).
Use `scripts/bench.sh`, `scripts/relay_bench.sh`, and `scripts/ab_bench.sh` on the actual deployment
host before making a performance claim or configuring deadline admission.
