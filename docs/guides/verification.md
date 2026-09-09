# Verification

## Release gate

```{mermaid}
flowchart LR
  C[Checkpoint] --> N[Native build]
  C --> P[PyTorch reference]
  N --> T[Unit + process tests]
  P --> U[ULP / relative error]
  T --> G[Install consumer + demos]
  U --> G
  G --> B[Budgets + hot-path allocations]
```

| Gate | Checks |
|---|---|
| C++ tests | Kernels, state, contracts, Relay, workers, process boundaries |
| Reference parity | Same checkpoint/input within configured error limits |
| Package | Installed `FlowEdge::Core` and `FlowEdge::Relay` compile and run |
| Runtime | Samples, traces, replay, metrics, migration, cancellation |
| Performance | Benchmarks, size budgets, setup allocations, zero hot-path allocations |

## Run everything

```bash
FLOWEDGE_BUILD_DIR=build-all \
  ./scripts/verify_all.sh models/mamba_flow.safetensors
cat build-all/budget-report.md
```

The script builds Core, Relay, tests, benchmarks, examples, process demos, install consumer, and
optional Python checks when dependencies are available.

## Focused checks

| Need | Command |
|---|---|
| C++ tests | `ctest --test-dir build --output-on-failure` |
| Formatting/static analysis | `FLOWEDGE_BUILD_DIR=build ./scripts/lint.sh` |
| PyTorch parity | `python scripts/verify_ulp.py models/mamba_flow.safetensors` |
| External head | `python scripts/verify_external_head.py build` |
| Relay lifecycle | `FLOWEDGE_BUILD_DIR=build ./scripts/relay_demo.sh models/mamba_flow.safetensors` |
| Benchmarks | `FLOWEDGE_BUILD_DIR=build ./scripts/bench.sh` |
| QoS | `./build/flowedge_job_qos_bench 1000000` |

## Invariants

| Invariant | Evidence |
|---|---|
| Older generations cannot publish | Scheduler and cancellation tests |
| Migration is exact and corruption-safe | Capsule + Mamba cross-engine tests |
| Draining cannot strand accepted work | Worker-drain tests |
| Failed lanes require explicit recovery | Quarantine tests |
| Trace bytes are portable | Canonical little-endian assertions |
| Event/metric storage is bounded | Fixed-capacity tests |
| Hot paths allocate zero | Relay and cooperative benchmarks |
| Installed API works | `test/install_consumer` |

Any new kernel, protocol, adapter, event type, or exporter needs a focused test and a runnable
verification entry when it adds a public surface.
