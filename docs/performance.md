# Performance

Use the numbers below as reproducible references, not deployment guarantees. Always rerun on the
target CPU, checkpoint, solver, thread count, and thermal profile.

## Headline results

Smoke checkpoint, one CPU thread unless noted; lower latency is better.

| Path | FlowEdge | Reference | Result |
|---|---:|---:|---:|
| Mamba forward, Windows | 0.091 ms | PyTorch 1.650 ms | 18.1× |
| Mamba forward, Linux | 0.071 ms | PyTorch 0.990 ms | 13.9× |
| Action p99, Windows | 0.579 ms | PyTorch 2.277 ms | 3.93× |
| Action p99, Linux | 0.650 ms | PyTorch 1.695 ms | 2.61× |
| Relay synchronous p99, Windows/Linux | 47.60 / 40.10 µs | — | reference |
| Action publish, Windows/Linux | 49.13 / 60.75 ns | — | reference |

## Runtime invariants

```{mermaid}
flowchart LR
  Load[Load model] --> Setup[Allocate weights + arenas]
  Setup --> Run[Repeated inference]
  Run --> Check{Heap allocation?}
  Check -->|zero| Pass[Contract holds]
  Check -->|non-zero| Fail[Regression: investigate]
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
| Core library | 300,670 bytes | 350,000 |
| Relay library | 422,170 bytes | 450,000 |
| Model setup allocations | 13 | 32 |
| Measured model hot-path allocations | 0 | 0 |
| Relay/cooperative hot-path allocations | 0 | 0 |

Model loading may allocate. The zero-allocation contract starts after engine/worker setup.

## Diffusion Policy reference

Pinned `lerobot/diffusion_pusht` conversion on the documented WSL2 host, four workers:

| Work | p50 |
|---|---:|
| One denoiser | 365.4 ms |
| 10-step DDIM sample | 3.765 s |

Dense-kernel optimization ranges from 1.09× to 3.18× over the earlier implementation. The fixed
LeRobot head is functional, but these numbers are not real-time claims.

## Relay and cooperative paths

| Benchmark | Windows | Linux |
|---|---:|---:|
| Relay pool, 1 worker | 37,240 req/s | 55,719 req/s |
| Relay pool, 2 workers | 73,187 req/s | 109,507 req/s |
| Migrate and finish | 478.56 ns/job | 292.08 ns/job |
| Direct route | 274.80 ns/job | 88.94 ns/job |
| EDF + lifecycle metrics | 3.49 µs/job | 1.19 µs/job |
| Shared-memory job service | 4.60 µs/job | 1.78 µs/job |

## Allocation-conscious APIs

| Path | Use |
|---|---|
| C++ Mamba scan | Strided B/C rows; no per-layer projection copies |
| Python decode | `decode_state()` exports directly to `bytes` |
| Python diffusion | `sample_diffusion_into(condition, noise, output, ...)` |
| LeRobot | `predict_action_chunk_into(...)`, `select_action_into(...)` |
| Rollout loop | Reuse noise and action buffers |

## Reproduce

<details class="fe-explorer">
<summary>Core tests and lint</summary>

```bash
FLOWEDGE_BUILD_DIR=build ./scripts/build.sh Release \
  -DFLOWEDGE_RELAY=ON -DFLOWEDGE_TESTS=ON -DFLOWEDGE_BENCH=ON
ctest --test-dir build --output-on-failure
FLOWEDGE_BUILD_DIR=build ./scripts/lint.sh
```
</details>

<details class="fe-explorer">
<summary>Complete release gate</summary>

```bash
FLOWEDGE_BUILD_DIR=build-all FLOWEDGE_VERIFY_BENCH_ITERS=500 \
  ./scripts/verify_all.sh models/mamba_flow.safetensors
cat build-all/budget-report.md
```

The gate runs tests, samples, Relay/job demos, benchmarks, install-consumer checks, and allocation
budgets.
</details>

<details class="fe-explorer">
<summary>Focused latency commands</summary>

```bash
./build/flowedge_engine_bench --benchmark_min_time=0.2s
./build/flowedge_latency_bench euler 20000
./build/flowedge_mamba_stream_bench models/mamba_flow.safetensors 1000
./build/flowedge_cooperative_job_bench 1000000
```
</details>

See [verification](guides/verification) for release gates and [allocation issue #63](https://github.com/elprofesoriqo/FlowEdge/issues/63)
for the stricter caller-owned load API.
