# Deadline profile

`flowedge-profile` turns a checkpoint into a repeatable control-loop gate. It warms the runtime,
measures external-condition flow inference, reports tail latency and hot-path allocations, and
returns a non-zero status when the configured period is missed.

```{mermaid}
flowchart LR
  C[Checkpoint] --> W[Warmup]
  W --> M[Timed samples]
  M --> S[p50 p95 p99 p999 max]
  S --> G{Budget + baseline}
  G -->|pass| P[Exit 0]
  G -->|fail| F[CI failure]
```

## Run it

```bash
cmake -S . -B build-profile -DCMAKE_BUILD_TYPE=Release -DFLOWEDGE_BENCH=ON
cmake --build build-profile --target flowedge-profile
./build-profile/flowedge-profile models/mamba_flow.safetensors \
  --solver euler --steps 10 --period-us 1000 --iterations 1000 \
  --json profile.json
```

| Exit | Meaning |
|---:|---|
| `0` | Profile passed the deadline and optional regression gate |
| `2` | Invalid arguments or profile |
| `3` | Selected deadline metric exceeded `--period-us` |
| `4` | p99 regression exceeded `--max-p99-regression` |
| `5` | Baseline environment is incompatible |

## Baselines in CI

```bash
./build-profile/flowedge-profile models/mamba_flow.safetensors \
  --period-us 1000 --compare benchmarks/baseline.json \
  --max-p99-regression 5 --json profile.json
```

Profiles use schema version `1` and include the checkpoint fingerprint, solver/NFE, precision,
compiler, build type, CPU label, affinity label, allocation count, all latency statistics, and
the selected budget result. Baselines are rejected when checkpoint, runtime, or host metadata differ;
use `--allow-environment-mismatch` only when that comparison is intentional. Set
`FLOWEDGE_BENCH_CPU` to record a stable CPU model label in CI.

The allocation count covers the timed calls only. Model loading, arena setup, and benchmark-buffer
construction happen before the measurement window; a zero result preserves FlowEdge's runtime
contract.
