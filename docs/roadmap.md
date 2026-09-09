# Roadmap

## Current release surface

| Layer | Status |
|---|---|
| Core | Mamba, flow head, Diffusion Policy head, CPU kernels, C/C++/Python APIs |
| State | Resumable solves, Mamba snapshots, model-bound job capsules |
| Relay | EDF, cancellation, worker drain, QoS, recovery, action delivery |
| Observability | Portable traces and fixed-memory metrics |
| Integrations | LeRobot deployment seam; external encoder boundary |

## Next in order

```{mermaid}
flowchart LR
  A[Release hardening] --> B[Fuzz + soak + protocol compatibility]
  B --> C[Checkpoint preflight + replay capsule]
  C --> D[Partner pilot: LeRobot]
  D --> E[Optional ROS 2 / Zenoh adapters]
```

| Priority | Deliverable | Exit condition |
|---:|---|---|
| 1 | Release hardening | Fuzzing, soak tests, compatibility and security checks |
| 2 | Checkpoint preflight | CI report for schema, precision, dimensions, arena, unsupported tensors |
| 3 | Replay capsule | Model/build/timing/input/output artifact replays in CI and on robots |
| 4 | LeRobot pilot | One reproducible rollout on supported hardware with safety ownership explicit |
| 5 | External adapters | Add only after a stable local contract and trace evidence |

## Parallel tracks

| Track | Candidate | Rule |
|---|---|---|
| Backbone | Transformer + KV cache | Keep fixed-shape and allocation contract |
| Weights | INT8, NUMA replication | Accept only with target-hardware measurements |
| Backend | CUDA, Tenstorrent | Backend-specific work stays behind kernel contracts |
| Perception | External encoder integrations | Keep encoders outside Core |

Do not expand FlowEdge into training, datasets, perception, arbitrary graph execution, or cluster
scheduling.
