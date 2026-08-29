# Overview

FlowEdge is a stack of small layers. Each layer has one job. Weights and scratch live in one arena.

## Layers

```{mermaid}
%%{init: {'theme':'base','flowchart':{'htmlLabels':false,'nodeSpacing':28,'rankSpacing':34,'useMaxWidth':false},'themeVariables':{'primaryColor':'#f6ead0','primaryBorderColor':'#7b2733','lineColor':'#7b2733','primaryTextColor':'#2b2521','secondaryColor':'#eaddbf','tertiaryColor':'#faf3e2','fontFamily':'system-ui, -apple-system, Segoe UI, Roboto, sans-serif','fontSize':'13px'}}}%%
graph TD
  API["C-ABI (fe_engine_*)"]
  API --> RT[Runtime]
  RT --> BB[Backbone]
  BB --> HD[Head]
  BB --> KER["Kernels"]
  HD --> KER
  LD[Loader] --> AR[Arena]
  AR --> RT
  AR --> BB
  AR --> HD
```

- Arena. Fixed bump allocator. Holds all weights and scratch. No malloc on the hot path.
- Loader. Zero-dependency safetensors mmap. Reads F32 and BF16.
- Runtime. `src/core/runtime/engine_runtime.*` owns arena-carved persistent state, model assembly,
  and the SPMC thread pool. The C ABI remains a narrow adapter in `src/core/api/`.
- Kernels. One backend-agnostic interface. CPU today. CUDA and Tenstorrent-shaped backends behind the same calls.
- Backbone. Compresses the prefix into a conditioning vector. Mamba today.
- Head. Turns the vector into an action. Flow matching today.
- API. The only public surface. A C-ABI over an opaque handle.

The head is also a first-class entry point. An external encoder can supply the conditioning vector
directly, and a checkpoint can contain only head tensors. This preserves the small runtime while
allowing FlowEdge to sit inside a heterogeneous ML stack.

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

For deadline-aware execution, the caller may replace the final head call with
`flow_begin(condition, noise)` followed by bounded `flow_advance(step_budget)` calls. See
[Cooperative execution](cooperative-execution).

## Why this way

A robot control loop runs at a fixed rate. The engine must return inside the period every cycle, not on average. So FlowEdge is built for worst-case latency, not throughput. Every design choice removes a source of variance.

Three rules carry that goal.

- Model code calls `matmul`, `discretize_and_scan`, and small span operations, never a device API. A backend port stays behind the kernel boundary.
- The footprint is computed at load and never changes. Weights, scratch, persistent decode state, worker storage, and the task ring are all carved from the arena.
- The runtime is hand assembled. There is no graph interpreter and no dynamic dispatch, so the instruction stream of a step is fixed and the branch and cache behavior repeat cycle to cycle.

The cost is manual work per architecture. A general graph runtime would remove that work but reintroduce dynamic allocation and dispatch, which is exactly the variance a control loop cannot absorb. See [ADR 0005](../decisions/0005-no-graph-runtime).
