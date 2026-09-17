# Performance

Intel i7-9750H (6C/12T), Release. Lower latency is better. Generated
checkpoints and fixed-token C++ benchmark fixtures are deliberately excluded
from this page: they do not establish robot-policy latency or control quality.

## Diffusion Policy

### Matched visual-policy replay

The {download}`captured CPU replay artifact <../bench/artifacts/policy/diffusion-pusht-cpu-replay.md>`
and its [raw samples](../bench/artifacts/policy/diffusion-pusht-cpu-replay.json) use the pinned
trained checkpoint, source RGB encoder/normalization, a two-observation PushT history,
and matched noise with ten DDIM steps. On this Windows/Clang host, the 20-sample
`threads=1` run measured 889 ms FlowEdge versus 1.448 s LeRobot median
preprocessing-to-chunk latency (policy p50 851 ms vs 1.409 s). FlowEdge is
1.66× faster on this packed, k-major-upsample replay. The removed generated-weight
flow-head fixture is not evidence about this policy.
Maximum absolute action error was 3.05e-5 in dataset units. The small run characterizes
this fixture, not stable p99, general task success, or robot suitability. See the artifact
for raw timings, shared-process memory accounting, versions, and unmeasured allocations.

`flowedge_diffusion_breakdown` times kernels at the U-Net's real shapes and native
`fe_engine_sample_diffusion` at explicit pool sizes. Caller-only native DDIM is 810 ms;
two workers 714 ms; four workers 567 ms; six workers 565 ms. The wall is DRAM: the
2048×L4 GEMM is 4.4 ms on one thread and 2.9 ms on four. A load-time `[K][OC][IC]`
copy cuts upsample-1024 from ~15 ms to ~1 ms. GroupNorm and Mish stay under 10 ms.
`tools/benchmark/export_kernel_mix.py` writes that table to `data/` as json/csv/parquet
for `data/explore.py`.

These replays load `flowedge.Engine` in-process. They do not start Relay.

### Thread sweep

Same checkpoint, processor, noise, and DDIM schedule. `threads=1` is the 20-sample
headline above. `threads=2` and `threads=4` are 10-sample runs that also give PyTorch
the same `torch.set_num_threads` budget. Artifacts:
{download}`threads=2 <../bench/artifacts/policy/diffusion-pusht-cpu-replay-threads2.md>`,
{download}`threads=4 <../bench/artifacts/policy/diffusion-pusht-cpu-replay-threads4.md>`.

| Threads | FlowEdge policy p50 | LeRobot policy p50 | FlowEdge / LeRobot | Native DDIM p50 |
|---:|---:|---:|---:|---:|
| 1 | 851 ms | 1409 ms | 0.60× | 810 ms (pool 0) |
| 2 | 800 ms | 1031 ms | 0.78× | 714 ms |
| 4 | 633 ms | 819 ms | 0.77× | 567 ms |
| 6 | — | — | — | 565 ms |

Four workers is the knee on this 6-core laptop. Extra SMT threads do not move the
denoiser. The fair single-thread compare remains `threads=1`.

### Kernel diagnostics

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
cmake --build build-diffusion-perf --target flowedge_kernels_bench -j2
./build-diffusion-perf/flowedge_kernels_bench \
  --benchmark_filter='BM_diffusion_' --benchmark_min_time=0.2s \
  --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
```

The former C++ diffusion-latency runner embedded zero conditions and generated
noise, so it was removed rather than presented as policy evidence. The matched
visual-policy replay above is the sole public end-to-end diffusion measurement.
These component figures are not a real-time guarantee.

Dense Conv1D now packs columns through the existing `matmul` when the caller
supplies a workspace of `conv1d_workspace_floats` (and `IC*K >= 32`).
ConvTranspose1D packs per kernel tap when `IC >= 32`. The published 8.55×
replay still used the scalar path on one thread; do not treat packing as a
beaten-PyTorch result until that artifact is regenerated.

### Four-thread replay

The published replay pinned `threads=1` for both FlowEdge and PyTorch. Re-run
the same fixture with a declared thread budget on both sides:

```bash
python tools/benchmark/run_policy_report.py models/diffusion_pusht.flowedge.safetensors \
  --source models/diffusion_pusht --revision 84a7c23178445c6bbf7e1a884ff497017910f653 \
  --steps 10 --iterations 20 --warmup 5 --threads 4 \
  --output bench/artifacts/policy/diffusion-pusht-cpu-replay-4t.json
```

PyTorch will also thread. Until that JSON exists, the public number remains the
one-thread artifact above. Kernel microbenchmarks with four workers (3.30 ms /
11.0 ms / 41.3 ms on the 512/1024/2048 Conv1D cases) are not policy latency.

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
[LeRobot adapter guide](guides/lerobot.md).

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

### Reproducible Relay audit

Use one Release build and repeat the script on the same host before accepting a
hot-path change. It records OS, commit, commands, timings, and allocation
counters for worker-pool, queue, QoS, streaming, drain, and action delivery.

```bash
FLOWEDGE_BUILD_DIR=build-relay \
FLOWEDGE_RELAY_BENCH_REPORT_DIR=bench-results \
./scripts/relay_bench.sh models/mamba_flow.safetensors
```

Windows PowerShell produces the same report format from a native Release build:

```powershell
$env:FLOWEDGE_BUILD_DIR="build-relay"
$env:FLOWEDGE_BUILD_JOBS="2" # use a bounded build to avoid memory pressure
$env:FLOWEDGE_RELAY_BENCH_REPORT_DIR="bench-results"
.\scripts\relay_bench.ps1 models\mamba_flow.safetensors
```

Keep the before/after report files with the change. Every reported
`hot_path_allocations` value must remain zero. The runners are audit tools, not
cross-host comparisons: compare medians only within a same-host, same-workload
series.

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
`bench/config/budgets.json`:

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
may still allocate today; a stricter caller-owned load API is not part of the
current product path.
