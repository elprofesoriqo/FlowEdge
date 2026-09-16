# Roadmap

## Current stack

| Layer | Complete |
|---|---|
| Core | Mamba, flow head, CPU kernels, BF16 weights, C/C++/Python APIs, cached-VLM SmolVLA action-expert boundary |
| State | Resumable flow solving, Mamba snapshots, generic job capsules |
| Relay service | Action and generic-job daemons, EDF, cancellation, administration |
| Action delivery | Multi-rate chunks, timed replacement, freshness and safety gate |
| Generic jobs | Contracts, IPC, bounded routing/admission, production Mamba streaming |
| Observability | Portable action/job traces; fixed-memory action/job metrics |

## Active development sequence

FlowEdge prioritizes a faster LeRobot Diffusion Policy drop-in, a SmolVLA
cached-expert plugin, then ARM/Jetson evidence. CPU Mamba remains a supported
CI fixture. Tenstorrent and DeadlineFlow stay research tracks.

| Order | Deliverable | Acceptance evidence |
|---:|---|---|
| 1 | Packed Diffusion Policy convolution | Replay at or below LeRobot/PyTorch; zero hot-path alloc |
| 2 | `flowedge` / `flowedge_smolvla` plugin | Seeded chunks, hold/drop/raise, hybrid expert parity |
| 3 | ARM/Jetson rollout JSON | Same schema as the Windows replay |
| 4 | Exact SmolVLA cache contract | Pinned checkpoint preflight and captured-VLM expert replay |
| 5 | One Tenstorrent vertical slice | One device, fixed shapes, persistent buffers |
| 6 | DeadlineFlow research evaluation | Quality/deadline curves after a user policy wins |

The first DeadlineFlow selector and CPU DDIM bridge are experimental. Full SmolVLA and
Tenstorrent execution remain unimplemented; the cached-VLM action expert is an explicit
boundary rather than a full-policy claim. The thesis delivers the measured backend;
the paper tests scheduling; the CppCon talk explains measured C++ design decisions.
See [policy evaluation](guides/policy-evaluation), [DeadlineFlow](guides/deadline-flow),
and the [Tenstorrent program](tenstorrent-program).

## Deferred expansion

These tracks require a concrete deployment need before taking priority over the active
sequence. Release correctness remains mandatory throughout.

| Track | Planned |
|---|---|
| Heads | ACT, VQ-BeT, π0; Diffusion Policy `diffusion_pusht` is done |
| Backbone | Transformer policy adapter and source SmolVLA encoder/action-expert parity |
| Weight traffic | NUMA replication experiments, INT8 |
| Backends | CUDA, Tenstorrent |
| Perception | External observation-encoder integration |
| Additional backbones | Mamba-3 after the active SmolVLA/Tenstorrent program |
| Transport | ROS 2, Zenoh, and distributed scheduling after deployment evidence |
