# Performance

Intel i7-9750H (6C/12T), Release. Comparisons use one CPU thread and PyTorch 2.13 CPU. Lower latency
is better.

## FlowEdge vs PyTorch

Measured 2026-08-31.

### Backbone forward

Same smoke checkpoint and token input `[1, 2, 3, 4]`.

| Platform | FlowEdge | PyTorch | Speedup |
|---|---:|---:|---:|
| Windows | 0.091 ms | 1.650 ms | 18.1x |
| Linux | 0.071 ms | 0.990 ms | 13.9x |

Linux:

```bash
FLOWEDGE_MODEL=models/mamba_flow.safetensors ./build/flowedge_engine_bench --benchmark_min_time=0.2s --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
python3 scripts/torch_ref.py bench models/mamba_flow.safetensors 1
```

Windows PowerShell:

```powershell
$env:FLOWEDGE_MODEL="models/mamba_flow.safetensors"
.\build\flowedge_engine_bench.exe --benchmark_min_time=0.2s --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
.\.venv\Scripts\python.exe scripts\torch_ref.py bench models/mamba_flow.safetensors 1
```

### Euler action head

Same deterministic 4-layer head, Euler 10 steps, and 20,000 iterations.

| Platform | FlowEdge mean | PyTorch mean | Speedup | FlowEdge p99 | PyTorch p99 | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| Windows | 375.73 us | 1,384.08 us | 3.68x | 579.40 us | 2,277.40 us | 3.93x |
| Linux | 395.18 us | 939.67 us | 2.38x | 649.80 us | 1,694.64 us | 2.61x |

Linux:

```bash
./build/flowedge_latency_bench euler 20000
python3 scripts/torch_ref.py latency euler 20000
```

Windows PowerShell:

```powershell
.\build\flowedge_latency_bench.exe euler 20000
.\.venv\Scripts\python.exe scripts\torch_ref.py latency euler 20000
```

## Diffusion Policy

Measured 2026-09-09 on the same Intel i7-9750H WSL2 host with GCC 13.3,
Release builds, and the pinned `lerobot/diffusion_pusht` revision
`84a7c23178445c6bbf7e1a884ff497017910f653`. The converted head is 1,006,060,612
bytes and loads with the fixed workspace; repeated inference reported zero heap
allocations.

The dominant dense-kernel microbenchmarks use the scalar backend, four
repetitions, and five Google Benchmark repetitions. The loop-interchanged,
fixed-shape implementation is compared with the pre-optimization implementation
on the same host:

| Case | Before p50 | Current p50 | Current speedup |
|---|---:|---:|---:|
| Conv1D 512x512, K5, L16 | 29.3 ms | 9.21 ms | 3.18x |
| Conv1D 1024x1024, K5, L8 | 57.9 ms | 31.6 ms | 1.83x |
| Conv1D 2048x2048, K5, L4 | 118 ms | 108 ms | 1.09x |
| Strided Conv1D 512x512, K3, L16→8 | 9.61 ms | 5.78 ms | 1.66x |
| ConvTranspose1D 512x512, K4, L8→16 | 10.5 ms | 6.00 ms | 1.75x |

The four-worker versions of the three dense cases measured 3.30 ms, 11.0 ms,
and 41.3 ms respectively. The benchmark host and commands are:

```bash
cmake -S . -B build-diffusion-perf -DCMAKE_BUILD_TYPE=Release \
  -DFLOWEDGE_BACKEND=cpu -DFLOWEDGE_BENCH=ON
cmake --build build-diffusion-perf --target flowedge_kernels_bench flowedge_diffusion_latency_bench -j2
./build-diffusion-perf/flowedge_kernels_bench \
  --benchmark_filter='BM_diffusion_' --benchmark_min_time=0.2s \
  --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
FLOWEDGE_THREADS=4 FLOWEDGE_BENCH_CPU='Intel-i7-9750H-WSL2' \
  ./build-diffusion-perf/flowedge_diffusion_latency_bench \
  /tmp/diffusion_pusht.flowedge.safetensors 10 3
```

For the public checkpoint, one denoiser was 365.4 ms p50 and a 10-step DDIM
sample was 3.765 s p50 with four workers. Caller-thread-only measurements were
1.041 s and 1.023 s respectively. These are reference measurements, not a
real-time guarantee; the next performance milestone is ISA-specialized
convolution or a packed-im2col path.

### Allocation-conscious inference paths

Mamba prefill and streaming decode now pass the projected B/C rows directly to
the scan kernel with a row stride. This removes two per-layer temporary buffers
and their row copies while preserving the existing zero-allocation hot-path
contract. `DiscretizeAndScan.SupportsStridedInputRows` covers the padded-row
layout used by the model.

The Python extension keeps one NumPy module handle per engine and exports decode
snapshots directly into `bytes`. Diffusion and LeRobot callers can use
`sample_diffusion_into`, `predict_action_chunk_into`, and `select_action_into`
with caller-owned buffers to avoid per-step output allocations. These APIs are
documented in [the Python API guide](api/python.md) and the
[LeRobot adapter guide](guides/lerobot).

## Relay

Measured 2026-08-31. Smoke checkpoint, Heun 6 steps, 5,000 requests.

| Benchmark | Windows | Linux |
|---|---:|---:|
| Synchronous p99 | 47.60 us | 40.10 us |
| Pool, 1 worker | 37,240 req/s | 55,719 req/s |
| Pool, 2 workers | 73,187 req/s | 109,507 req/s |

Linux:

```bash
./build/flowedge_relay_bench models/mamba_flow.safetensors 5000 0
./build/flowedge_relay_pool_bench models/mamba_flow.safetensors 5000 1 0
./build/flowedge_relay_pool_bench models/mamba_flow.safetensors 5000 2 0
```

Windows PowerShell:

```powershell
.\build\flowedge_relay_bench.exe models/mamba_flow.safetensors 5000 0
.\build\flowedge_relay_pool_bench.exe models/mamba_flow.safetensors 5000 1 0
.\build\flowedge_relay_pool_bench.exe models/mamba_flow.safetensors 5000 2 0
```

## Cooperative jobs

Measured 2026-09-01. One million tiny counter jobs; framework overhead only.

| Benchmark | Windows | Linux |
|---|---:|---:|
| Migrate and finish | 478.56 ns/job | 292.08 ns/job |
| Direct route, run, and return | 274.80 ns/job | 88.94 ns/job |
| EDF pool + lifecycle metrics | 3.49 us/job | 1.19 us/job |
| Shared-memory client/service + pool + metrics | 4.60 us/job | 1.78 us/job |

Linux:

```bash
./build/flowedge_cooperative_job_bench 1000000
```

Windows PowerShell:

```powershell
.\build\flowedge_cooperative_job_bench.exe 1000000
```

### Deadline queue dispatch

Measured 2026-09-02. 32 queued jobs, 3,000 rounds, median of repeated runs.

| Platform | Before | Current | Speedup |
|---|---:|---:|---:|
| Windows | 4.62 us/job | 1.36 us/job | 3.40x |
| Linux | 2.85 us/job | 1.22 us/job | 2.33x |

Linux:

```bash
./build/flowedge_job_queue_bench 3000
```

Windows PowerShell:

```powershell
.\build\flowedge_job_queue_bench.exe 3000
```

### Mamba streaming adapter

Measured 2026-09-02. Four tokens, one thread, median of five 1,000-job runs.

| Platform | Direct adapter | Generic route | Migrate after token 2 + finish |
|---|---:|---:|---:|
| Windows | 108.16 us/job | 109.54 us/job | 114.32 us/job |
| Linux | 115.74 us/job | 115.85 us/job | 129.32 us/job |

Linux:

```bash
./build/flowedge_mamba_stream_bench models/mamba_flow.safetensors 1000
```

Windows PowerShell:

```powershell
.\build\flowedge_mamba_stream_bench.exe models\mamba_flow.safetensors 1000
```

### Worker drain handoff

Measured 2026-09-02. Eight work units, drain after the first boundary, median of five 10,000-job
runs.

| Platform | Mean | p50 | p99 |
|---|---:|---:|---:|
| Windows | 34.24 us | 5.80 us | 144.20 us |
| Linux | 52.41 us | 54.54 us | 108.81 us |

Linux:

```bash
./build/flowedge_worker_drain_bench 10000
```

Windows PowerShell:

```powershell
.\build\flowedge_worker_drain_bench.exe 10000
```

### Action delivery

Measured 2026-09-02. Four-step, two-axis chunks; median of five 1,000,000-iteration runs.

| Platform | Accept + publish / step | Replace + blended publish |
|---|---:|---:|
| Windows | 49.13 ns | 225.63 ns |
| Linux | 60.75 ns | 306.73 ns |

Linux:

```bash
./build/flowedge_action_delivery_bench 1000000
```

Windows PowerShell:

```powershell
.\build\flowedge_action_delivery_bench.exe 1000000
```

### QoS overload admission

Measured 2026-09-03. Typed best-effort rejection with protected interactive and critical slots;
median of five 1,000,000-iteration runs.

| Platform | Rejection latency |
|---|---:|
| Windows | 114.481 ns |
| Linux | 65.6609 ns |

Linux:

```bash
for i in 1 2 3 4 5; do ./build/flowedge_job_qos_bench 1000000; done
```

Windows PowerShell:

```powershell
1..5 | ForEach-Object { .\build\flowedge_job_qos_bench.exe 1000000 }
```

These results are references, not deployment guarantees.

## Size and initialization budgets

The complete verification script also reports the checked-in budgets in
`bench/budgets.json`:

```bash
FLOWEDGE_BUILD_DIR=build-all ./scripts/verify_all.sh models/mamba_flow.safetensors
cat build-all/budget-report.md
```

The report measures Core and Relay static-library bytes and model initialization
allocations. Initialization is checked across three `load -> run -> free` cycles
so a setup regression or lifecycle mismatch is visible. It fails when a checked-in
limit is exceeded and separately rejects any allocation in the measured model hot
path. The limits are review thresholds for the Release build, not claims about
every compiler or linker configuration.

FlowEdge's hard allocation guarantee applies after engine/worker initialization:
repeated inference and Relay hot paths must allocate zero heap memory. Model loading
may still allocate today; strict zero-allocation initialization is tracked separately
because it requires caller-owned storage or a load-time arena API; see
[issue #63](https://github.com/elprofesoriqo/FlowEdge/issues/63).
