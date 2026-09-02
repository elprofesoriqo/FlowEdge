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

These results are references, not deployment guarantees.
