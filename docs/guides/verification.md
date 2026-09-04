# Verification

## Release gates

| Gate | Proves |
|---|---|
| Kernel/unit tests | C++ kernels match independent references |
| PyTorch parity | Same checkpoint/input stays within ULP and relative-error limits |
| Relay process test | Real client, shared memory, daemon, deadline result, shutdown |
| Generic process test | Real child service, typed success/rejection, backpressure, shutdown |
| Worker-pool tests | Parallel lanes, freshness, QoS, rolling drain, quarantine, recovery |
| Action-delivery tests | Chunk shape, replacement, timing, freshness, bounds, delta limits |
| Mamba adapter tests | Exact migrated output and real-model generic routing |
| Job observability tests | Fixed event capacity, kinds, progress, migration, timings, exporters |
| Trace round-trip | Canonical bytes for action and generic job records |
| Install consumer | Installed `FlowEdge::Core` and `FlowEdge::Relay` configure, link, run |

```{mermaid}
flowchart LR
  C[Checkpoint] --> F[FlowEdge]
  C --> P[PyTorch reference]
  F --> V[ULP + relative error]
  P --> V
  V --> T[Unit + process + package gates]
```

## Run everything

Linux:

```bash
./scripts/verify_all.sh models/mamba_flow.safetensors
```

Windows PowerShell with Git Bash available:

```powershell
bash scripts/verify_all.sh models/mamba_flow.safetensors
```

The script builds Core, Relay, tests, benchmarks, every C++ example, generic-job JSONL inspection,
the Relay lifecycle demo, action replay, all metric formats, installation, and a downstream consumer.
Python checks run when their dependencies are available.

## Focused commands

| Need | Command |
|---|---|
| C++ tests | `ctest --test-dir build --output-on-failure` |
| PyTorch parity | `python scripts/verify_ulp.py models/mamba_flow.safetensors` |
| External-head + streaming smoke | `python scripts/verify_external_head.py build` |
| Relay lifecycle | `FLOWEDGE_BUILD_DIR=build ./scripts/relay_demo.sh models/mamba_flow.safetensors` |
| Formatting/static analysis | `FLOWEDGE_BUILD_DIR=build ./scripts/lint.sh` |
| Benchmarks | `FLOWEDGE_BUILD_DIR=build ./scripts/bench.sh` |
| Relay benchmarks | `FLOWEDGE_BUILD_DIR=build ./scripts/relay_bench.sh` |
| QoS overload | `./build/flowedge_job_qos_bench 1000000` |

## Key invariants

| Invariant | Coverage |
|---|---|
| Older generations cannot publish as current | Scheduler, head-pool, job-pool cancellation tests |
| Full result rings cannot lose work | `JobTransport.PreservesTypedResultsAcrossOutputBackpressure` |
| Admission includes active and queued lanes | Multi-lane EDF tests |
| Migration is exact and corruption-safe | Cooperative capsule and Mamba cross-engine tests |
| Draining cannot strand accepted work | Worker drain handoff and queued-work tests |
| Lower service classes cannot consume reserved slots | QoS reservation and equal-deadline tests |
| A failed lane cannot silently rejoin | Quarantine and explicit-recovery tests |
| Trace bytes are portable | Representative little-endian byte assertions |
| Event overflow is bounded | `JobEvents.BuffersValidatedLifecycleRecordsWithoutGrowth` |
| Job metrics keep fixed kinds | `JobMetrics.RecordsKindsProgressPreemptionMigrationAndLatency` |
| Hot paths allocate nothing | Relay and cooperative-job benchmarks |
| Queue depth does not multiply clock reads | Deadline queue benchmark |
| Live handoff remains bounded | Worker drain benchmark |
| Controller publication stays bounded | Action delivery benchmark and multi-rate example |
| Public API works after install | `test/install_consumer` |

## Install consumer

```bash
cmake --install build --prefix build/install-check
cmake -S test/install_consumer -B build/install-consumer \
  -DCMAKE_PREFIX_PATH="$PWD/build/install-check"
cmake --build build/install-consumer -j
./build/install-consumer/flowedge_install_consumer
```

New kernels, protocols, adapters, event types, and exporters require a focused test plus inclusion in
`verify_all.sh` when they add a runnable surface.
