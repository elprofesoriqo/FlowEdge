# Roadmap

```{image} _static/figures/sequence.svg
:alt: convert, CPU replay, Jetson or SO-100, same contracts
:class: fe-fig
```

## Current stack

| Layer | Complete |
|---|---|
| Core | Mamba, flow head, CPU Transformer decoder fixture, CPU kernels, BF16 weights, C/C++/Python APIs, cached-VLM SmolVLA action-expert boundary |
| State | Resumable flow solving, Mamba snapshots, generic job capsules |
| Relay service | Action and generic-job daemons, EDF, cancellation, administration |
| Action delivery | Multi-rate chunks, timed replacement, freshness and safety gate |
| Generic jobs | Contracts, IPC, bounded routing/admission, production Mamba streaming |
| Observability | Portable action/job traces; fixed-memory action/job metrics |

## Active development sequence

FlowEdge prioritizes a measured LeRobot policy on CPU/ARM hardware, then SmolVLA expert
quality, then Tenstorrent. CPU Mamba + flow stays a complete included path.

| Order | Deliverable | Acceptance evidence |
|---:|---|---|
| 1 | Transformer issue #10 | Kernel gtests, tiny-gpt2 conversion JSON, optional latency JSON |
| 2 | Correct visual Diffusion Policy deployment | Processor/encoder/history parity, seeded chunks |
| 3 | Matched replay and a period loop | PyTorch p50 plus `--period-ms` miss counts |
| 4 | One Jetson or SO-100 recorded loop | Issue #70 JSON; limits stay in the adapter |
| 5 | Cached SmolVLA expert contract | Captured-VLM replay; VLM remains in LeRobot |
| 6 | One Tenstorrent vertical slice | One device, fixed shapes, persistent buffers |
| 7 | DeadlineFlow research | Quality vs miss rate; not a shipping backend |

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
