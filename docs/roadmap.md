# Roadmap

## Current stack

| Layer | Complete |
|---|---|
| Core | Mamba, flow head, CPU kernels, BF16 weights, C/C++/Python APIs |
| State | Resumable flow solving, Mamba snapshots, generic job capsules |
| Relay service | Shared memory, EDF, cancellation, worker pool, placement |
| Generic jobs | Contracts, process IPC, routing, bounded lanes, per-kind admission |
| Observability | Portable action/job traces; fixed-memory action/job metrics |

## Next sequence

```{mermaid}
flowchart LR
  M[Streaming Mamba adapter] --> G[Worker draining + migration]
  G --> A[Action overlap + safety gate]
  A --> R[ROS 2 / Zenoh adapters]
```

| Order | Milestone | Why now |
|---:|---|---|
| 1 | Streaming Mamba job adapter | First production model on the generic process contract |
| 2 | Worker draining and live migration | Enables maintenance and failure recovery |
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
