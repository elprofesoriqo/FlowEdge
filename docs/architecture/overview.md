# Overview

FlowEdge is a stack of small layers. Each layer has one job. Weights and scratch live in one arena.

## Layers

```{mermaid}
%%{init: {'theme':'base','flowchart':{'htmlLabels':false,'nodeSpacing':28,'rankSpacing':34,'useMaxWidth':false},'themeVariables':{'primaryColor':'#f6ead0','primaryBorderColor':'#7b2733','lineColor':'#7b2733','primaryTextColor':'#2b2521','secondaryColor':'#eaddbf','tertiaryColor':'#faf3e2','fontFamily':'system-ui, -apple-system, Segoe UI, Roboto, sans-serif','fontSize':'13px'}}}%%
graph TD
  API["C-ABI (fe_engine_*)"]
  API --> BB[Backbone]
  BB --> HD[Head]
  BB --> KER["Kernels"]
  HD --> KER
  LD[Loader] --> AR[Arena]
  AR --> BB
  AR --> HD
```

- Arena. Fixed bump allocator. Holds all weights and scratch. No malloc on the hot path.
- Loader. Zero-dependency safetensors mmap. Reads F32 and BF16.
- Kernels. One backend-agnostic interface. CPU today. CUDA and Tenstorrent behind the same calls.
- Backbone. Compresses the prefix into a conditioning vector. Mamba today.
- Head. Turns the vector into an action. Flow matching today.
- API. The only public surface. A C-ABI over an opaque handle.

## The matrix

FlowEdge grows on two axes. Backbones and heads share the same kernels and converter.

```{mermaid}
%%{init: {'theme':'base','flowchart':{'htmlLabels':false,'nodeSpacing':28,'rankSpacing':34,'useMaxWidth':false},'themeVariables':{'primaryColor':'#f6ead0','primaryBorderColor':'#7b2733','lineColor':'#7b2733','primaryTextColor':'#2b2521','secondaryColor':'#eaddbf','tertiaryColor':'#faf3e2','fontFamily':'system-ui, -apple-system, Segoe UI, Roboto, sans-serif','fontSize':'13px'}}}%%
graph LR
  M["Mamba (done)"] --> F["Flow (done)"]
  M --> D[Diffusion]
  TR[Transformer] --> ACT[ACT]
  TR --> P[pi0]
```

The first model of a family pays for its kernels. The rest reuse them.

## Data flow

```{mermaid}
%%{init: {'theme':'base','flowchart':{'htmlLabels':false,'nodeSpacing':28,'rankSpacing':34,'useMaxWidth':false},'themeVariables':{'primaryColor':'#f6ead0','primaryBorderColor':'#7b2733','lineColor':'#7b2733','primaryTextColor':'#2b2521','secondaryColor':'#eaddbf','tertiaryColor':'#faf3e2','fontFamily':'system-ui, -apple-system, Segoe UI, Roboto, sans-serif','fontSize':'13px'}}}%%
sequenceDiagram
  participant U as Caller
  participant E as Engine
  participant B as Backbone
  participant H as Head
  U->>E: sample(tokens, noise, steps)
  E->>B: run prefix
  B-->>E: conditioning vector
  E->>H: integrate ODE from noise
  H-->>E: action chunk
  E-->>U: action
```

## Why this way

A robot control loop runs at a fixed rate. The engine must return inside the period every cycle, not on average. So FlowEdge is built for worst-case latency, not throughput. Every design choice removes a source of variance.

Three seams carry that goal.

- The kernel interface is the portability seam. Model code calls `matmul` and `selective_scan`, never a device API, so a CPU to GPU port is a link-time swap, not a rewrite.
- The arena is the memory seam. The footprint is computed at load and never changes, so there is no allocator call, no fragmentation, and no first-touch page fault during a step.
- Hand assembly is the latency seam. There is no graph interpreter and no dynamic dispatch, so the instruction stream of a step is fixed and the branch and cache behavior repeat cycle to cycle.

The cost is manual work per architecture. A general graph runtime would remove that work but reintroduce dynamic allocation and dispatch, which is exactly the variance a control loop cannot absorb. See [ADR 0005](../decisions/0005-no-graph-runtime).
