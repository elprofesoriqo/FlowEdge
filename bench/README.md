# Benchmark layout

```text
bench/
├── kernels/     dense-kernel and threaded-matmul microbenchmarks
├── runtime/     Relay, queue, deadline, delivery, and streaming benchmarks
├── config/      checked-in performance budgets
└── artifacts/   portable reports grouped by policy, transformer, and SmolVLA
```

| Area | Measures | Typical command |
|---|---|---|
| `kernels/` | kernel throughput and threading overhead | `scripts/bench.sh` |
| `runtime/` | queueing, worker, deadline, delivery, and Relay tail behavior | `scripts/relay_bench.ps1` or `scripts/verify_all.sh` |
| `artifacts/policy/` | matched LeRobot/PyTorch versus FlowEdge replay | `tools/benchmark/run_policy_report.py` |
| `artifacts/transformer/` | real checkpoint parity and embedding-boundary checks | `tools/verification/verify_transformer_reference.py` |
| `artifacts/smolvla/` | checkpoint inspection and support/preflight evidence | `tools/smolvla_preflight.py` and `flowedge-inspect` |

The `artifacts/` tree contains evidence, not benchmark source. Each report must
state the model revision, host, build, parameters, raw samples, and limitations.
Synthetic C++ policy fixtures are intentionally not part of this tree.
