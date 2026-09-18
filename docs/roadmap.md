# Roadmap

```{image} _static/figures/sequence.svg
:alt: convert, CPU replay, Jetson or SO-100, same contracts
:class: fe-fig
```

## Current stack

| Layer | Complete |
|---|---|
| Core | Mamba, flow head, CPU Transformer decoder baseline, CPU kernels, BF16 weights, C/C++/Python APIs, cached-VLM SmolVLA action-expert boundary |
| State | Resumable flow solving, Mamba snapshots, generic job capsules |
| Relay service | Action and generic-job daemons, EDF, cancellation, administration |
| Action delivery | Multi-rate chunks, timed replacement, freshness and safety gate |
| Generic jobs | Contracts, IPC, bounded routing/admission, production Mamba streaming |
| Observability | Portable action/job traces; fixed-memory action/job metrics |

## Active development sequence

FlowEdge keeps Diffusion Policy and flow matching as the product path. The
Transformer decoder is a fixture for later VLA work. CUDA is the next accelerator,
not Tenstorrent-first. Jetson/ARM (#70) still needs a board JSON.

| Order | Deliverable | Acceptance evidence |
|---:|---|---|
| 1 | Transformer issue #10 | Kernel gtests, tiny-gpt2 conversion JSON, optional latency JSON |
| 2 | CUDA linking TU | `FLOWEDGE_BACKEND=cuda` configures only with `nvcc`; ISA kernels link ([#160](https://github.com/elprofesoriqo/FlowEdge/issues/160)) |
| 3 | Device-resident flow head | Persistent device buffers; no per-op host round-trip ([#161](https://github.com/elprofesoriqo/FlowEdge/issues/161)) |
| 4 | Diffusion Policy on CUDA | `diffusion_ops` on device; DDIM sample vs CPU reference ([#162](https://github.com/elprofesoriqo/FlowEdge/issues/162)) |
| 5 | Matched GPU replay | Same observations vs PyTorch CUDA ([#163](https://github.com/elprofesoriqo/FlowEdge/issues/163)); then LeRobot CUDA ([#164](https://github.com/elprofesoriqo/FlowEdge/issues/164)) |
| 6 | Jetson / ARM replay | Device JSON for [#70](https://github.com/elprofesoriqo/FlowEdge/issues/70); x86 JSON is not ARM evidence |
| 7 | Tenstorrent / π0 / ACT | After CUDA evidence; see deferred table |

The first DeadlineFlow selector and CPU DDIM bridge are experimental. Full SmolVLA
and Tenstorrent execution remain unimplemented. See
[policy evaluation](guides/policy-evaluation), [DeadlineFlow](guides/deadline-flow),
[CUDA](architecture/cuda), and the [Tenstorrent program](tenstorrent-program).

## Deferred expansion

These tracks require a concrete deployment need before taking priority over CUDA
flow and Diffusion Policy.

| Track | Planned |
|---|---|
| Heads | ACT, VQ-BeT, π0; Diffusion Policy `diffusion_pusht` is done on CPU |
| Backbone | Transformer policy adapter and source SmolVLA encoder |
| Weight traffic | NUMA replication experiments, INT8 |
| Backends | Device-resident CUDA; Tenstorrent TT-Metal |
| Perception | External observation-encoder integration |
| Additional backbones | Mamba-3 after CUDA policy evidence |
| Transport | ROS 2, Zenoh, and distributed scheduling after deployment evidence |
