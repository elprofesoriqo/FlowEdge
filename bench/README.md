# Benchmark layout

![do not mix the evidence](../docs/_static/figures/benches.svg)

```text
bench/
├── kernels/     dense-kernel and threaded matmul
├── runtime/     Relay, queue, deadline, delivery
├── config/      checked-in budgets
└── artifacts/   policy, transformer, SmolVLA reports
```

| Area | Measures | Command |
|---|---|---|
| `kernels/` | kernel cost | `scripts/bench.sh` |
| `runtime/` | admission, queues, allocations | `scripts/relay_bench.sh` |
| `artifacts/policy/` | FlowEdge vs LeRobot/PyTorch | `python -m flowedge_dev bench policy` |
| `artifacts/smolvla/` | cached-expert / hybrid | `python -m flowedge_dev verify smolvla` |
| `artifacts/transformer/` | GPT-2 incubator parity | `python -m flowedge_dev verify transformer` |

Artifacts are evidence, not sources. Each report names host, revision, and limits. Synthetic policy fixtures are not in this tree.
