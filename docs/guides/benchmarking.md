# Benchmarking and regression review

FlowEdge has four distinct performance tools. Use the latency benchmark to validate a control-loop
deadline, the kernel benchmark to isolate CPU-kernel changes, and the engine benchmark to measure a
real checkpoint's backbone forward pass. When Relay is enabled, its benchmark measures the full
producer-to-consumer local inference path. They answer different questions and should not be combined
into one headline number.

## Establishing a baseline

Configure a Release build with benchmarks and record the exact host, compiler, checkpoint digest,
solver, affinity, and power mode beside its JSON output. Leave the host idle and use the same
environment for every A/B comparison.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFLOWEDGE_BENCH=ON
cmake --build build --config Release -j
./build/flowedge_kernels_bench --benchmark_out=build/kernels.json --benchmark_out_format=json
```

For the optional Relay path:

```bash
FLOWEDGE_RELAY_BENCH_ITERS=500 ./scripts/relay_bench.sh models/mamba_flow.safetensors
```

Set `FLOWEDGE_THREADS=0..8` to pin the Core worker count. Otherwise the Relay benchmark uses Core's
bandwidth-aware automatic choice.

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

## Interpreting a result

A threshold failure begins an investigation; it is not enough evidence to blame a patch by itself.
Repeat the run, confirm the CPU frequency and affinity, then look at the affected kernel's source and
its FLOP/s and weight-bandwidth counters. A run on another operating system, processor, compiler, or
power profile requires a fresh baseline. Keep both JSON artifacts with a performance-sensitive pull
request so a reviewer can reproduce the comparison without rebuilding the original revision.
