# FlowEdge

<img class="fe-hero-img" src="_static/hero.png" alt="FlowEdge" />

<p class="fe-lede">A compact C++23 runtime for predictable robotics-policy inference on CPU.</p>

| At a glance | Current boundary |
|---|---|
| Runtime | Mamba backbone, flow head, fixed LeRobot Diffusion Policy head |
| Memory | Static arena; supported hot paths allocate zero after setup |
| Interfaces | C ABI, C++/CMake, Python, optional Relay service |
| CPU | Scalar, AVX2, NEON; FP32 and BF16 weights |
| Scope | Inference and deployment contracts, not training or perception |

## Pick a path

| You want to… | Open |
|---|---|
| Run the included policy | [Getting started](getting-started) |
| Feed an external encoder | [Capabilities](capabilities) |
| Use Python buffers without per-step output allocation | [Python API](api/python) |
| Connect LeRobot | [LeRobot guide](guides/lerobot) |
| Serve local clients with deadlines | [Relay quickstart](guides/relay-quickstart) |
| Move state between workers | [Cooperative jobs](guides/cooperative-jobs) |
| Check a checkpoint before deployment | [Preflight](guides/checkpoint-preflight) |
| Reproduce a latency or allocation result | [Performance](performance) |
| Publish a LeRobot comparison | [Edge benchmarks](guides/edge-benchmarks) |

## Runtime flow

```{mermaid}
flowchart LR
  W[.safetensors] --> L[Loader]
  L --> A[Fixed arena]
  T[Tokens / condition] --> B[Mamba or external head]
  A --> B
  B --> H[Flow / diffusion solver]
  H --> O[Action chunk]
  O --> G[Optional Relay safety gate]
  G --> R[Robot controller]
```

## Available vs planned

| Area | Available | Planned |
|---|---|---|
| Backbone | Mamba SSM | Transformer + KV cache |
| Head | Flow matching; fixed LeRobot Diffusion Policy | ACT, VQ-BeT, DiT, Pi0 |
| Solver | Euler, Heun, RK4; DDIM, DDPM | — |
| Backend | CPU scalar / AVX2 / NEON | CUDA, Metal, Vulkan, Tenstorrent |
| Deployment | C/C++/Python; local Relay; traces and metrics | ROS 2, Zenoh, external runtime adapters |

> **Not a training framework, graph runtime, perception stack, distributed scheduler, or safety controller.**

```{toctree}
:hidden:
:caption: Start
Overview <self>
getting-started
capabilities
performance
```

```{toctree}
:hidden:
:caption: Architecture
architecture/overview
architecture/memory
architecture/loader
architecture/deployment-profile
architecture/kernels
architecture/backbones
architecture/heads
architecture/cooperative-execution
architecture/model-porting
decisions/index
```

```{toctree}
:hidden:
:caption: API
api/c-abi
api/python
```

```{toctree}
:hidden:
:caption: Guides
guides/add-a-head
guides/add-a-backbone
guides/converter
guides/checkpoint-preflight
guides/edge-benchmarks
guides/diffusion-policy
guides/lerobot
guides/sanitizers
guides/verification
guides/observability
guides/cooperative-jobs
guides/relay-quickstart
guides/action-delivery
guides/generic-job-daemon
```

```{toctree}
:hidden:
:caption: Reference
roadmap
product-direction
FlowEdge Relay <ecosystem/relay-proposal>
```
