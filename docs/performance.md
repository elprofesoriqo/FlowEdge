# Performance

CPU. `threads=1` is the fair compare. Lower latency is better.

```{image} _static/figures/deadline.svg
:alt: 10 ms period vs 851 ms CPU DDIM, on-miss hold
:class: fe-fig
```

```{image} _static/perf.gif
:alt: Matched PushT replay, FlowEdge 851 ms vs PyTorch 1409 ms
:class: fe-fig
```

```{image} _static/figures/compare.svg
:alt: Policy p50 851 vs 1409 ms, threads=1
:class: fe-fig
```

```{image} _static/figures/vs-arch.svg
:alt: Flow matching ULP versus Diffusion Policy matched p50
:class: fe-fig
```

## Diffusion Policy

{download}`replay <../bench/artifacts/policy/diffusion-pusht-cpu-replay.md>`
· [JSON](../bench/artifacts/policy/diffusion-pusht-cpu-replay.json)
· {download}`threads=2 <../bench/artifacts/policy/diffusion-pusht-cpu-replay-threads2.md>`
· {download}`threads=4 <../bench/artifacts/policy/diffusion-pusht-cpu-replay-threads4.md>`
· {download}`period 10 ms hold <../bench/artifacts/policy/diffusion-pusht-cpu-period10-hold.md>`

| | FlowEdge | LeRobot / PyTorch |
|---|---:|---:|
| Policy p50 | **851 ms** | **1409 ms** |
| Preprocess-to-chunk p50 | 889 ms | 1448 ms |
| Max abs action error | 3.05e-5 | — |
| FlowEdge / LeRobot | 0.60× | |

| Threads | FlowEdge policy p50 | LeRobot policy p50 | FlowEdge / LeRobot | Native DDIM p50 |
|---:|---:|---:|---:|---:|
| 1 | 851 ms | 1409 ms | 0.60× | 810 ms (pool 0) |
| 2 | 800 ms | 1031 ms | 0.78× | 714 ms |
| 4 | 633 ms | 819 ms | 0.77× | 567 ms |
| 6 | — | — | — | 565 ms |

| Period | On miss | Misses | Step p50 / p99 / max |
|---|---|---:|---|
| 10 ms | hold | 20 / 20 | 655 / 812 / 834 ms |

| Case | Before p50 | Current p50 | Speedup | 4 workers |
|---|---:|---:|---:|---:|
| Conv1D 512×512, K5, L16 | 29.3 ms | 9.21 ms | 3.18× | 3.30 ms |
| Conv1D 1024×1024, K5, L8 | 57.9 ms | 31.6 ms | 1.83× | 11.0 ms |
| Conv1D 2048×2048, K5, L4 | 118 ms | 108 ms | 1.09× | 41.3 ms |
| Strided Conv1D 512×512, K3, L16→8 | 9.61 ms | 5.78 ms | 1.66× | — |
| ConvTranspose1D 512×512, K4, L8→16 | 10.5 ms | 6.00 ms | 1.75× | — |

| Native DDIM | p50 |
|---|---:|
| pool 0 | 810 ms |
| 2 workers | 714 ms |
| 4 workers | 567 ms |
| 6 workers | 565 ms |

## CUDA Diffusion Policy

NVIDIA GeForce GTX 1650 is the headline, not `threads=1`. Same observation file as the CPU replay. Not TensorRT/ONNX. Not Jetson/ARM.

{download}`CUDA replay <../bench/artifacts/policy/diffusion-pusht-cuda-replay.md>`
· [JSON](../bench/artifacts/policy/diffusion-pusht-cuda-replay.json)

| | FlowEdge CUDA | PyTorch CUDA |
|---|---:|---:|
| Policy p50 | **420 ms** | **346 ms** |
| Preprocess-to-chunk p50 | 425 ms | 352 ms |
| Max abs action error | 4.58e-5 | — |
| FlowEdge / LeRobot | 1.21× | |

A 4GB card cannot hold both U-Nets; the runner frees FlowEdge before loading the PyTorch U-Net. PyTorch CUDA is faster on this device; the CPU `threads=1` 851 vs 1409 ms figure is unchanged.

### How to get them

```bash
python -m flowedge_dev bench policy models/diffusion_pusht.flowedge.safetensors \
  --source models/diffusion_pusht --threads 1

python -m flowedge_dev pipeline rollout models/diffusion_pusht.flowedge.safetensors \
  --steps 20 --threads 4 --period-ms 10 --on-miss hold

python -m flowedge_dev bench mix

cmake -S . -B build-diffusion-perf -DCMAKE_BUILD_TYPE=Release \
  -DFLOWEDGE_BACKEND=cpu -DFLOWEDGE_BENCH=ON
cmake --build build-diffusion-perf --target flowedge_kernels_bench -j2
./build-diffusion-perf/flowedge_kernels_bench \
  --benchmark_filter='BM_diffusion_' --benchmark_min_time=0.2s \
  --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
```

CUDA matched replay (WSL, `FLOWEDGE_BACKEND=cuda`, PyTorch CUDA):

```bash
python -m flowedge_dev bench policy models/diffusion_pusht.flowedge.safetensors \
  --source models/diffusion_pusht --revision 84a7c23178445c6bbf7e1a884ff497017910f653 \
  --steps 10 --iterations 10 --warmup 2 --threads 1 --device cuda \
  --observations bench/artifacts/policy/diffusion-pusht-cpu-replay.observations.npz \
  --build-dir /path/to/cuda-build \
  --output bench/artifacts/policy/diffusion-pusht-cuda-replay.json
```

## Flow matching

```{image} _static/flowedge.gif
:alt: Flow matching Euler steps from noise to action
:class: fe-fig
```

| | Value |
|---|---|
| Checkpoint | `models/mamba_flow.safetensors` |
| Max rel-error vs PyTorch | ~1e-6 |
| Rel-error gate | 2e-3 |
| ULP gate | 4096 |
| Prefix | `[1, 2, 3, 4]` (correctness, not a p50) |
| Matched policy p50 | — |

### How to get them

```bash
python -m flowedge_dev verify ulp models/mamba_flow.safetensors
./build/flow_sample models/mamba_flow.safetensors euler 10
```

## Relay

| Benchmark | Windows | Linux |
|---|---:|---:|
| Synchronous p99 | 47.60 us | 40.10 us |
| Pool, 1 worker | 37,240 req/s | 55,719 req/s |
| Pool, 2 workers | 73,187 req/s | 109,507 req/s |

### How to get them

```bash
./build/flowedge_relay_bench models/mamba_flow.safetensors 5000 0
./build/flowedge_relay_pool_bench models/mamba_flow.safetensors 5000 1 0
./build/flowedge_relay_pool_bench models/mamba_flow.safetensors 5000 2 0

FLOWEDGE_BUILD_DIR=build-relay \
FLOWEDGE_RELAY_BENCH_REPORT_DIR=bench-results \
./scripts/relay_bench.sh models/mamba_flow.safetensors
```

```powershell
.\build\flowedge_relay_bench.exe models/mamba_flow.safetensors 5000 0
.\build\flowedge_relay_pool_bench.exe models/mamba_flow.safetensors 5000 1 0
.\build\flowedge_relay_pool_bench.exe models/mamba_flow.safetensors 5000 2 0
$env:FLOWEDGE_BUILD_DIR="build-relay"
$env:FLOWEDGE_BUILD_JOBS="2"
$env:FLOWEDGE_RELAY_BENCH_REPORT_DIR="bench-results"
.\scripts\relay_bench.ps1 models\mamba_flow.safetensors
```

## Cooperative jobs

| Benchmark | Windows | Linux |
|---|---:|---:|
| Migrate and finish | 478.56 ns/job | 292.08 ns/job |
| Direct route, run, and return | 274.80 ns/job | 88.94 ns/job |
| EDF pool + lifecycle metrics | 3.49 us/job | 1.19 us/job |
| Shared-memory client/service + pool + metrics | 4.60 us/job | 1.78 us/job |

| Deadline queue | Before | Current | Speedup |
|---|---:|---:|---:|
| Windows | 4.62 us/job | 1.36 us/job | 3.40× |
| Linux | 2.85 us/job | 1.22 us/job | 2.33× |

| Mamba stream adapter | Direct | Generic route | Migrate after token 2 |
|---|---:|---:|---:|
| Windows | 108.16 us/job | 109.54 us/job | 114.32 us/job |
| Linux | 115.74 us/job | 115.85 us/job | 129.32 us/job |

| Worker drain | Mean | p50 | p99 |
|---|---:|---:|---:|
| Windows | 34.24 us | 5.80 us | 144.20 us |
| Linux | 52.41 us | 54.54 us | 108.81 us |

| Action delivery | Accept + publish / step | Replace + blended publish |
|---|---:|---:|
| Windows | 49.13 ns | 225.63 ns |
| Linux | 60.75 ns | 306.73 ns |

| QoS rejection | Latency |
|---|---:|
| Windows | 114.481 ns |
| Linux | 65.6609 ns |

### How to get them

```bash
./build/flowedge_cooperative_job_bench 1000000
./build/flowedge_job_queue_bench 3000
./build/flowedge_mamba_stream_bench models/mamba_flow.safetensors 1000
./build/flowedge_worker_drain_bench 10000
./build/flowedge_action_delivery_bench 1000000
for i in 1 2 3 4 5; do ./build/flowedge_job_qos_bench 1000000; done
```

```powershell
.\build\flowedge_cooperative_job_bench.exe 1000000
.\build\flowedge_job_queue_bench.exe 3000
.\build\flowedge_mamba_stream_bench.exe models\mamba_flow.safetensors 1000
.\build\flowedge_worker_drain_bench.exe 10000
.\build\flowedge_action_delivery_bench.exe 1000000
1..5 | ForEach-Object { .\build\flowedge_job_qos_bench.exe 1000000 }
```

## Budgets

```bash
FLOWEDGE_BUILD_DIR=build-all ./scripts/verify_all.sh models/mamba_flow.safetensors
cat build-all/budget-report.md
```
