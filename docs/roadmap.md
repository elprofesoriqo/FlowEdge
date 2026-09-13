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

## Active development sequence

FlowEdge prioritizes a validated trained LeRobot policy, SmolVLA execution on
Tenstorrent, then DeadlineFlow experiments. CPU Mamba remains a supported reference.

| Order | Deliverable | Acceptance evidence |
|---:|---|---|
| 1 | Correct visual Diffusion Policy deployment | Processor/encoder/history parity, seeded chunks, reset tests |
| 2 | Matched replay and task evaluation | Raw timings, checkpoint/processor hashes, PushT episode outcomes |
| 3 | Exact SmolVLA reference contract | Pinned checkpoint, token/mask/state interfaces, trajectory fixtures |
| 4 | One Tenstorrent vertical slice | One device, fixed shapes, persistent buffers, complete Euler integration |
| 5 | Accelerator optimization | Profile-driven layouts, traces or kernels; reproducible hardware records |
| 6 | DeadlineFlow research evaluation | Fixed/adaptive baselines, quality/deadline curves, underruns, ablations |
| 7 | Broader support | Second hardware generation/model after the first path passes its gates |

The first DeadlineFlow selector and CPU DDIM bridge are experimental. SmolVLA and
Tenstorrent execution remain unimplemented. The thesis delivers the measured backend;
the paper tests scheduling; the CppCon talk explains measured C++ design decisions.
See [policy evaluation](guides/policy-evaluation), [DeadlineFlow](guides/deadline-flow),
and the [Tenstorrent program](tenstorrent-program).

## Deferred expansion

These tracks require a concrete deployment need before taking priority over the active
sequence. Release correctness remains mandatory throughout.

| Track | Planned |
|---|---|
| Heads | ACT, VQ-BeT, π0; Diffusion Policy `diffusion_pusht` is done |
| Backbone | Transformer and KV cache |
| Weight traffic | NUMA replication experiments, INT8 |
| Backends | CUDA, Tenstorrent |
| Perception | External observation-encoder integration |
| Additional backbones | Mamba-3 after the active SmolVLA/Tenstorrent program |
| Transport | ROS 2, Zenoh, and distributed scheduling after deployment evidence |
