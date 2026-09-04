# Roadmap

## Current stack

| Layer | Complete |
|---|---|
| Core | Mamba, flow head, CPU kernels, BF16 weights, C/C++/Python APIs |
| State | Resumable flow solving, Mamba snapshots, generic job capsules |
| Relay service | Action and generic-job daemons, EDF, cancellation, administration |
| Action delivery | Multi-rate chunks, timed replacement, freshness and safety gate |
| Generic jobs | Contracts, IPC, bounded routing/admission, production Mamba streaming |
| Observability | Portable action/job traces; fixed-memory action/job metrics |

## Next sequence

```{mermaid}
flowchart LR
  M[Production Mamba adapter ✓] --> G[Worker drain + live migration ✓]
  G --> A[Action overlap + safety gate ✓]
  A --> D[Generic job daemon ✓]
  D --> S[Worker supervision + QoS complete]
  S --> R[Release hardening]
  R --> I[ROS 2 / Zenoh adapters]
```

| Order | Milestone | Why now |
|---:|---|---|
| 1 | Streaming Mamba job adapter | Done: exact resume and routed execution |
| 2 | Worker draining and live migration | Done: bounded rolling handoff |
| 3 | Action overlap, freshness, final safety gate | Done: allocation-free controller boundary |
| 4 | Standalone generic-job daemon | Done: separate data/admin planes and managed Mamba lanes |
| 5 | Worker supervision and QoS | Done: queue reservations, quarantine, explicit recovery |
| 6 | Release hardening | Next: fuzzing, soak tests, protocol compatibility, security boundary |
| 7 | ROS 2, Zenoh, inference-server adapters | Optional integrations over stable contracts |

## Parallel model/backend work

| Track | Planned |
|---|---|
| Heads | Diffusion Policy, ACT, VQ-BeT, π0 |
| Backbone | Transformer and KV cache |
| Weight traffic | NUMA replication experiments, INT8 |
| Backends | CUDA, Tenstorrent |
| Perception | External observation-encoder integration |
