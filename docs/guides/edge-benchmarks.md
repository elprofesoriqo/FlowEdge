# Reproducible policy reports

```{mermaid}
flowchart LR
  M[Model + immutable revision] --> X[Same processor and observation history]
  X --> T[LeRobot / PyTorch]
  X --> F[FlowEdge native]
  T & F --> J[Validated JSON]
  J --> R[Markdown report + raw samples]
```

## Run one report

```bash
python tools/benchmark/run_policy_report.py models/policy.safetensors \
  --source models/diffusion_pusht --revision <revision> \
  --steps 10 --iterations 100 --warmup 5 --threads 1 \
  --output bench/artifacts/policy/diffusion-pusht-report.json
```

The published replay used `--threads 1`. To compare a four-thread host, pass
`--threads 4` on the same command; PyTorch uses the same budget. Do not mix
thread counts across backends.

| Output | Includes |
|---|---|
| JSON | model/processor hashes, contract, host/build, raw samples, parity, limitations |
| Markdown | p50/p95/p99 table, FlowEdge/LeRobot ratios, exact commands |
| `.observations.npz` | fixed observation history used by the matched replay |

The reference is not a toy implementation: it is LeRobot running through
PyTorch. End-to-end includes preprocessing, observation history, encoder, and
iterative sampling; it excludes camera capture, IPC, and robot delivery.

## Contract

| Field | Rule |
|---|---|
| Quantiles | `p50 <= p95 <= p99` |
| Backend status | `measured`, `not_measured`, `unavailable`, or `blocked` |
| Missing values | Keep them null and explain the reason; never write zero |
| Reproducibility | Pin revision, hashes, host, compiler, threads, steps, warmup, and iterations |

Validate an existing artifact with:

```bash
python tools/benchmark/benchmark_artifact.py \
  bench/artifacts/policy/diffusion-pusht-report.json \
  --report bench/artifacts/policy/diffusion-pusht-report.md
```

Small runs are smoke checks, not stable p99 evidence. Repeat on the target
machine and publish the JSON, report, and raw samples together.
