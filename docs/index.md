# FlowEdge

<img class="fe-hero-img" src="_static/hero.png" alt="FlowEdge" />

<p class="fe-lede">FlowEdge is a small C++23 engine for running flow-matching action policies on the edge. It compiles to a dependency-free static binary and runs the whole control loop without a single heap allocation.</p>

## What it is

A policy has two parts. A backbone reads the observation prefix and compresses it into a conditioning vector. A head takes that vector, starts from a noise sample, and integrates it into an action chunk. FlowEdge runs both on the CPU, fast enough for a real-time loop.

It is not a training framework and not a graph runtime. It is a fixed set of hand-written architectures that share one kernel library. Every weight and every scratch buffer comes from a single arena, sized once at load, so the runtime path allocates nothing.

## Purpose

Training frameworks trade latency for flexibility. PyTorch runs an interpreter over each op, and ONNX Runtime carries a graph engine and a stack of dependencies. Neither is built for a loop that has to finish inside a fixed period on an embedded board. FlowEdge is. It stays small, keeps its memory static, and is checked against PyTorch in CI on every commit.

## What is implemented

<div class="fe-grid">
  <div class="fe-card">
    <h4>Backbones</h4>
    <ul><li class="done">Mamba SSM</li><li>Transformer</li></ul>
  </div>
  <div class="fe-card">
    <h4>Heads</h4>
    <ul><li class="done">Flow matching</li><li>Diffusion Policy</li><li>DiT</li></ul>
  </div>
  <div class="fe-card">
    <h4>Solvers</h4>
    <ul><li class="done">Euler</li><li class="done">Heun</li><li class="done">RK4</li></ul>
  </div>
  <div class="fe-card">
    <h4>Precision</h4>
    <ul><li class="done">FP32</li><li class="done">BF16</li><li>INT8</li></ul>
  </div>
  <div class="fe-card">
    <h4>Backends</h4>
    <ul><li class="done">CPU (AVX2 / NEON)</li><li>CUDA</li><li>Tenstorrent</li></ul>
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

## Start here

- [Getting Started](getting-started). Build, load a model, sample an action.
- [Architecture](architecture/overview). How the layers fit and why.
- [C-ABI](api/c-abi). The public surface.

```{toctree}
:hidden:
:caption: Start
Overview <self>
getting-started
comparison
```

```{toctree}
:hidden:
:caption: Architecture
architecture/overview
architecture/memory
architecture/loader
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
guides/verification
guides/benchmarking
```

```{toctree}
:hidden:
:caption: Reference
roadmap
product-direction
ecosystem/relay-proposal
```
