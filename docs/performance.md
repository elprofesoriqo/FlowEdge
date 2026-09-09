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

| Metric | Current report | Budget |
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
