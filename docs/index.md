# FlowEdge

<img class="fe-hero-img" src="_static/hero.png" alt="FlowEdge" />

<p class="fe-lede">FlowEdge is a small C++23 engine for running flow-matching and diffusion action policies on the edge. It compiles to a dependency-free static binary and runs the whole control loop without a single heap allocation.</p>

## What it is

A policy has two parts. A backbone reads the observation prefix and compresses it into a conditioning vector. A head takes that vector, starts from a noise sample, and integrates it into an action chunk. FlowEdge runs both on the CPU, fast enough for a real-time loop.

It is not a training framework and not a graph runtime. It is a fixed set of hand-written architectures that share one kernel library. Every weight and every scratch buffer comes from a single arena, sized once at load, so the runtime path allocates nothing.

## What you can use today

| Goal | Entry point |
|---|---|
| Run a complete Mamba + flow policy | [Getting Started](getting-started) |
| Keep an existing encoder and use only the action head | [Capabilities](capabilities) |
| Run deadline-aware inference between processes | [Relay quickstart](guides/relay-quickstart) |
| Deliver safe multi-rate action chunks | [Action delivery](guides/action-delivery) |
| Run managed streaming jobs | [Generic job daemon](guides/generic-job-daemon) |
| Migrate streaming or custom model state | [Cooperative jobs](guides/cooperative-jobs) |
| Compare FlowEdge with PyTorch | [Performance](performance) |
| Convert or port a supported checkpoint | [Converter](guides/converter) |
| Deploy a supported LeRobot policy | [LeRobot adapter](guides/lerobot) |
| Preflight a checkpoint before deployment | [Checkpoint preflight](guides/checkpoint-preflight) |

## Repository map

| Area | Contents |
|---|---|
| `src/core/` | Allocation-free model and solver runtime |
| `src/relay/` | Optional local scheduling and transport |
| `python/` and `integrations/lerobot/` | Python API and LeRobot deployment adapter |
| `convert/` and `tools/` | Checkpoint conversion and inspection |
| `examples/` | Small C++ and Python entry-point samples |
| `scripts/` | Build, test, verification, and maintainer benchmarks |
| `bench/` | Focused Google Benchmark targets, enabled with `FLOWEDGE_BENCH=ON` |
| `cmake/` | Installed-package export template; generated `CMakeFiles/` is ignored |

## Implementation status

<div class="fe-grid">
  <div class="fe-card">
    <h4>Backbones</h4>
    <ul><li class="done">Mamba SSM</li><li>Transformer (planned)</li></ul>
  </div>
  <div class="fe-card">
    <h4>Heads</h4>
    <ul><li class="done">Flow matching</li><li class="done">Diffusion Policy</li><li>DiT (experimental)</li></ul>
  </div>
  <div class="fe-card">
    <h4>Solvers</h4>
    <ul><li class="done">Euler</li><li class="done">Heun</li><li class="done">RK4</li><li class="done">DDIM / DDPM</li></ul>
  </div>
  <div class="fe-card">
    <h4>Precision</h4>
    <ul><li class="done">FP32</li><li class="done">BF16</li><li>INT8 (planned)</li></ul>
  </div>
  <div class="fe-card">
    <h4>Backends</h4>
    <ul><li class="done">CPU (AVX2 / NEON)</li><li>CUDA (planned)</li><li>Tenstorrent (planned)</li></ul>
  </div>
  <div class="fe-card">
    <h4>Interfaces</h4>
    <ul><li class="done">C-ABI</li><li class="done">Python</li><li class="done">Converter</li></ul>
  </div>
</div>

## How it fits

```{mermaid}
%%{init: {'theme':'base','flowchart':{'htmlLabels':false,'nodeSpacing':28,'rankSpacing':34,'useMaxWidth':false},'themeVariables':{'primaryColor':'#f6ead0','primaryBorderColor':'#7b2733','lineColor':'#7b2733','primaryTextColor':'#2b2521','secondaryColor':'#eaddbf','tertiaryColor':'#faf3e2','fontFamily':'system-ui, -apple-system, Segoe UI, Roboto, sans-serif','fontSize':'13px'}}}%%
graph TD
  W[".safetensors"] --> L[Loader]
  L --> A[Arena]
  T[tokens] --> B[Backbone]
  A --> B
  B --> C[conditioning vector]
  C --> H[Head]
  H --> ACT[action chunk]
  K["Kernels: CPU"] -.-> B
  K -.-> H
```

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
guides/diffusion-policy
guides/lerobot
guides/edge-benchmarks
guides/deadline-profile
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
LeRobot edge-inference RFC draft <ecosystem/lerobot-rfc>
```
