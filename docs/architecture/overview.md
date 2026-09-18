# Overview

FlowEdge is a small stack with one job per layer. The point is worst-case latency on a robot period, not average throughput.

A general graph runtime would accept more architectures with less hand work. It would also reintroduce allocation, dispatch, and a working set that is hard to bound — exactly the variance a control loop cannot absorb. So the model is hand-assembled: convert tensors into known names, carve one arena, call kernels through one header. [ADR 0005](../decisions/0005-no-graph-runtime).

```{image} ../_static/figures/purpose.svg
:alt: Hardware runs the loop, FlowEdge runs the policy, LeRobot trains and drives the robot
:class: fe-fig
```

```{image} ../_static/figures/big-flow.svg
:alt: Whole FlowEdge flow from checkpoint to motors
:class: fe-fig
```

```{image} ../_static/figures/relay.svg
:alt: Optional Relay flow from sensor or VLA through shared-memory rings to Core
:class: fe-fig
```

```{image} ../_static/figures/stack.svg
:alt: Layers from plugin down to the arena
:class: fe-fig
```

Two product heads share that stack:

- **Flow matching** — a short ODE \(dx/dt = v(x,t \mid c)\) from noise to action. Mamba (or a caller-supplied encoder) produces \(c\).
- **Diffusion Policy** — DDIM on a Conv1D U-Net. LeRobot’s RGB encoder stays outside Core.

Transformer conversion and a cached SmolVLA expert are incubators, not those heads.

```{image} ../_static/figures/matrix.svg
:alt: Mamba plus flow matching, or LeRobot encoder plus DP U-Net
:class: fe-fig
```

```{image} ../_static/figures/dataflow.svg
:alt: tokens or condition, backbone, flow ODE, action chunk
:class: fe-fig
```

- **Arena.** One bump slab. Load may allocate it; the hot path does not.
- **Loader.** Zero-dep `.safetensors` mmap. F32 and BF16.
- **Runtime.** `engine_runtime` owns persistent state, model assembly, SPMC pool. C ABI is a thin adapter.
- **Kernels.** One header. CPU today. `FLOWEDGE_BACKEND=cuda` keeps Mamba, flow, and Diffusion Policy heads on device. Vulkan / Tenstorrent (Metal) stay planned.
- **Backbone.** Prefix → condition. Mamba today (constant-size state). Transformer is a CPU decoder fixture, not a policy.
- **Head.** Condition + noise → action. Flow matching or Diffusion Policy. An external encoder can skip the backbone.

Split the ODE with `flow_begin` / `flow_advance` if the robot has to yield. [Cooperative execution](cooperative-execution). Model code calls `matmul` and `discretize_and_scan`, never a device API. Footprint is computed at load.
