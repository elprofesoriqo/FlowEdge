# Current performance reference

These are dated local reference measurements for the current repository, not portable product
guarantees. Use them to understand what each benchmark measures and to check that your own results are
in the expected order of magnitude.

## Test environment

Measured on 2026-08-31 with an Intel Core i7-9750H (6 cores, 12 threads, 12 MiB L3), Release builds,
the repository's two-layer `models/mamba_flow.safetensors` smoke checkpoint, and no explicit affinity.

| Environment | Compiler | Notes |
|---|---|---|
| Windows 11 native | Clang 23.1.0 | Native executable, AVX2 CPU backend |
| Ubuntu 24.04 under WSL 2 | GCC 13.3.0 | Same physical host and checkpoint |

The machine was otherwise idle, but frequency, background services, virtualization, and laptop
thermals were not locked. Do not use these numbers directly as robot admission limits.

<div class="fe-explorer-grid">
<details class="fe-explorer" open>
<summary>Choosing a latency number</summary>
<p>Use Core latency for an in-process call and synchronous Relay p99 for an isolated local service.</p>
<p>Do not use pool backlog latency as a single-request deadline.</p>
</details>
<details class="fe-explorer">
<summary>Choosing a capacity number</summary>
<p>Use pool requests/second to compare worker counts under concurrency.</p>
<p>Re-run with the real model, affinity, thermal policy, and producer rate.</p>
</details>
<details class="fe-explorer">
<summary>Reading generic overhead</summary>
<p>The nanosecond rows use a tiny counter backend to expose framework cost.</p>
<p>Model compute, payload movement, and state size dominate production adapters.</p>
</details>
</div>

## Core execution

| Benchmark | Windows | WSL 2 | Allocation result |
|---|---:|---:|---:|
| Smoke-checkpoint backbone forward, median | 71 us | 71 us | Engine-managed fixed runtime memory |
| Synthetic Euler action, 10 steps, mean | 345.83 us | 331.53 us | 0 hot allocations |
| Synthetic Euler action, 10 steps, p99 | 555.20 us | 542.60 us | 0 hot allocations |
| Synthetic Euler action, 10 steps, p999 | 720.90 us | 696.00 us | 0 hot allocations |

`BM_engine_forward` used five repetitions with a 0.2-second minimum sample. The synthetic action
benchmark uses condition 768, action 32, hidden 256, time embedding 128, four layers, Euler, and 20,000
iterations. It intentionally measures a stable action-head shape independently of the small smoke
checkpoint.

## Relay execution

All Relay cases use the smoke checkpoint, Heun with six solver steps (12 NFE), caller-only Core
engines, 5,000 requests, and checksummed local shared memory.

| Path | Environment | Throughput | Mean latency | p99 latency | Hot allocations |
|---|---|---:|---:|---:|---:|
| Synchronous producer → EDF → worker → consumer | Windows | 39,318 req/s | 25.38 us | 47.60 us | 0 |
| Synchronous producer → EDF → worker → consumer | WSL 2 | 60,136 req/s | 16.59 us | 40.10 us | 0 |
| Bounded pool, 1 worker | Windows | 37,240 req/s | 1,734.29 us | 3,008.40 us | 0 |
| Bounded pool, 1 worker | WSL 2 | 55,719 req/s | 1,158.60 us | 1,583.20 us | 0 |
| Bounded pool, 2 workers | Windows | 73,187 req/s | 895.72 us | 1,206.60 us | 0 |
| Bounded pool, 2 workers | WSL 2 | 109,507 req/s | 597.86 us | 880.70 us | 0 |

The pool benchmark intentionally maintains a bounded EDF backlog, so its latency includes queueing
and should not be compared directly with the synchronous single-request latency. Its purpose is to
show capacity and scaling under concurrency. On this run, two workers delivered 1.97x Windows and
1.97x WSL throughput versus one worker while sharing one 986,112-byte immutable checkpoint store.

## Cooperative migration and routing overhead

`flowedge_cooperative_job_bench` runs three work units, exports a canonical checksummed capsule,
restores it into a second backend, and finishes five more units. A second loop validates a request,
looks up a frozen adapter registry, prepares and binds the backend, runs eight work units, and encodes
a typed result.

| Environment | Complete migrated job | Per migrated work unit | Complete routed job | Hot allocations |
|---|---:|---:|---:|---:|
| Windows | 179.23 ns | 22.40 ns | 85.86 ns | 0 |
| WSL 2 | 306.52 ns | 38.32 ns | 92.51 ns | 0 |

These are framework measurements around a tiny counter backend. The routed value includes registry
lookup, all eight trivial work units, and result creation; it is not just lookup latency. A real
model's compute, state, and payload copies will dominate, so benchmark the actual adapter and capsule
sizes.

## Reproduce the measurements

```bash
FLOWEDGE_MODEL=models/mamba_flow.safetensors \
  ./build/flowedge_engine_bench --benchmark_min_time=0.2s \
  --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
./build/flowedge_latency_bench euler 20000
./build/flowedge_relay_bench models/mamba_flow.safetensors 5000 0
./build/flowedge_relay_pool_bench models/mamba_flow.safetensors 5000 1 0
./build/flowedge_relay_pool_bench models/mamba_flow.safetensors 5000 2 0
./build/flowedge_cooperative_job_bench 1000000
```

For a deployment decision, record checkpoint digest, compiler, CPU, affinity, governor/power mode,
worker/thread configuration, temperature, sustained duration, p999/max latency, allocation count, and
rejection rate. See [Benchmarking and regression review](guides/benchmarking) for A/B comparison and
deadline calibration.
