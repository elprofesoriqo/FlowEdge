# Benchmarking and regression review

FlowEdge has several distinct performance tools. Use the latency benchmark to validate a control-loop
deadline, the kernel benchmark to isolate CPU-kernel changes, and the engine benchmark to measure a
real checkpoint's backbone forward pass. When Relay is enabled, its benchmark measures the full
producer-to-consumer local inference path. They answer different questions and should not be combined
into one headline number.

| Question | Tool |
|---|---|
| How fast is one supported backbone forward? | `flowedge_engine_bench` |
| What is action-head tail latency without model loading? | `flowedge_latency_bench` |
| Which kernel changed? | `flowedge_kernels_bench` |
| How does inner Core threading scale? | `flowedge_threaded_matmul_bench` |
| What does one complete local Relay request cost? | `flowedge_relay_bench` |
| How does the bounded worker pool scale? | `flowedge_relay_pool_bench` |
| What are generic capsule and registry-route costs? | `flowedge_cooperative_job_bench` |

See [Current performance reference](../performance) for dated results from the repository's validated
Windows/WSL host.

## Establishing a baseline

Configure a Release build with benchmarks and record the exact host, compiler, checkpoint digest,
solver, affinity, and power mode beside its JSON output. Leave the host idle and use the same
environment for every A/B comparison.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFLOWEDGE_BENCH=ON
cmake --build build --config Release -j
./build/flowedge_kernels_bench --benchmark_out=build/kernels.json --benchmark_out_format=json
```

`scripts/bench.sh` also runs the threaded-matmul worker scaling sweep and the matched FP32/BF16 model
latency benchmark when both checkpoints are available.

For the optional Relay path:

```bash
FLOWEDGE_RELAY_BENCH_ITERS=500 ./scripts/relay_bench.sh models/mamba_flow.safetensors
```

Set `FLOWEDGE_THREADS=0..8` to pin the Core worker count. Otherwise the Relay benchmark uses Core's
bandwidth-aware automatic choice.

The Relay output includes `p99_ns_per_nfe`, calculated from the end-to-end p99 request latency and
the solver's NFE count. It is a conservative starting value for `flowedge-relayd --nfe-ns`, not a
portable constant. Re-measure with the production checkpoint, thread count, affinity, power policy,
and sustained thermal load. Add `--admission-reserve-ns` for transport/controller jitter, then verify
rejection rates against captured traces before enforcing the policy in a physical control loop.

`flowedge_relay_pool_bench` keeps a bounded EDF backlog and measures the independent worker threads
under concurrent load. Compare one and two workers with caller-only Core engines before changing the
daemon configuration:

```bash
./build/flowedge_relay_pool_bench models/mamba_flow.safetensors 5000 1 0
./build/flowedge_relay_pool_bench models/mamba_flow.safetensors 5000 2 0
```

The last two arguments are outer model workers and Core background threads per worker. The benchmark
reports throughput, queue-inclusive latency, and hot allocations. `scripts/relay_bench.sh` runs both
the single-request transport benchmark and this pool benchmark; override its pool case with
`FLOWEDGE_RELAY_POOL_WORKERS` and `FLOWEDGE_RELAY_POOL_THREADS`.

More workers are useful only while throughput rises without violating tail latency, memory, or CPU
budgets. Workers share immutable checkpoint tensors but retain private scratch, transformed constants,
solver state, and decode state. Record `shared_weight_bytes`, private memory, and resident memory as
well as requests/second. Avoid multiplying outer workers by large inner Core thread pools without an
explicit oversubscription experiment.

The cooperative benchmark isolates framework and capsule overhead with a tiny backend:

```bash
./build/flowedge_cooperative_job_bench 1000000
```

It reports two loops: partial execution, canonical export, validation, restore, and completion; then
request validation, frozen-registry lookup, adapter prepare/bind, completion, and typed result
encoding. It fails when either measured path allocates. Real model state and results can be much
larger, so repeat with the intended adapter and payload sizes.

On Windows, use the `.exe` names from PowerShell. In WSL or Git Bash, the repository's Bash wrappers
are available as well.

## Comparing a Git baseline

`scripts/ab_bench.sh` creates a detached temporary worktree for the baseline and distinct build
directories for both revisions. It does not switch branches, stash files, reuse build output, or hide
benchmark failures. Google Benchmark takes repeated samples and emits mean CPU time for every kernel.

```bash
./scripts/ab_bench.sh --baseline-ref main --runs 9 --threshold 5
```

The command writes `baseline-kernels.json`, `candidate-kernels.json`, and `report.md` to a timestamped
directory under `bench-results/`. It exits non-zero if a benchmark is absent from the candidate or its
mean CPU time increases by more than the threshold. Restrict a run when investigating one operation:

```bash
./scripts/ab_bench.sh --filter 'BM_matmul_(in|out)_proj' --runs 15 --threshold 3
```

For an offline run, point `FLOWEDGE_FETCHCONTENT_SOURCE_ROOT` at an existing CMake `_deps` directory
that contains `mdspan-src` and `googlebenchmark-src`; both temporary builds then reuse those sources.
On Windows Git Bash, the report step falls back to WSL `python3` when no runnable native Python is available;
the benchmark executables themselves still build and run natively on Windows. The script launches
them through a clean PowerShell process so MSYS file descriptors cannot affect benchmark aggregation.

## Interpreting a result

A threshold failure begins an investigation; it is not enough evidence to blame a patch by itself.
Repeat the run, confirm the CPU frequency and affinity, then look at the affected kernel's source and
its FLOP/s and weight-bandwidth counters. A run on another operating system, processor, compiler, or
power profile requires a fresh baseline. Keep both JSON artifacts with a performance-sensitive pull
request so a reviewer can reproduce the comparison without rebuilding the original revision.
