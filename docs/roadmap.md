# Roadmap

## Current stack

| Layer | Complete |
|---|---|
| Core | Mamba, flow head, CPU kernels, BF16 weights, C/C++/Python APIs |
| State | Resumable flow solving, Mamba snapshots, generic job capsules |
| Relay service | Shared memory, EDF, cancellation, worker pool, placement |
| Generic jobs | Contracts, IPC, bounded routing/admission, production Mamba streaming |
| Observability | Portable action/job traces; fixed-memory action/job metrics |

## Next sequence

```{mermaid}
flowchart LR
  M[Production Mamba adapter ✓] --> G[Worker draining + live migration]
  G --> A[Action overlap + safety gate]
  A --> R[ROS 2 / Zenoh adapters]
```

| Order | Milestone | Why now |
|---:|---|---|
| 1 | Streaming Mamba job adapter | Done: exact resume and routed execution |
| 2 | Worker draining and live migration | Next: maintenance and failure recovery |
| 3 | Action overlap, freshness, final safety gate | Robotics-specific output policy |
| 4 | ROS 2, Zenoh, inference-server adapters | Optional integrations over stable contracts |

## Parallel model/backend work

| Track | Planned |
|---|---|
| Heads | Diffusion Policy, ACT, VQ-BeT, π0 |
| Backbone | Transformer and KV cache |
| Weight traffic | NUMA replication experiments, INT8 |
| Backends | CUDA, Tenstorrent |
| Perception | External observation-encoder integration |
