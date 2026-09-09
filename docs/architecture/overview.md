# Architecture overview

## Layers

```{mermaid}
flowchart TD
  API[C ABI / C++ / Python] --> RT[Runtime]
  RT --> BB[Backbone]
  RT --> HD[Head]
  BB --> K[Kernels]
  HD --> K
  Loader[Safetensors loader] --> Arena[Fixed arena]
  Arena --> RT
  Arena --> BB
  Arena --> HD
```

| Layer | Responsibility | Current implementation |
|---|---|---|
| Loader | Read F32/BF16 checkpoint tensors | Zero-dependency safetensors |
| Arena | Weights, scratch, persistent state | Fixed bump allocator |
| Runtime | Model assembly and lifecycle | C++23 engine + optional workers |
| Backbone | Prefix → condition | Mamba SSM |
| Head | Condition/noise → action | Flow matching; fixed Diffusion Policy |
| Kernels | Backend-neutral compute | CPU scalar / AVX2 / NEON |
| API | Stable integration boundary | C ABI, CMake, Python |

## Matrix

```{mermaid}
flowchart LR
  M[Mamba] --> F[Flow]
  M --> D[Diffusion Policy]
  T[Transformer: planned] --> A[ACT: planned]
  T --> P[Pi0: planned]
```

Backbones and heads share kernels and conversion contracts. An external encoder may provide the
condition directly; a head-only checkpoint is valid.

## Inference flow

```{mermaid}
sequenceDiagram
  participant U as Caller
  participant E as Engine
  participant B as Backbone
  participant H as Head
  U->>E: tokens / condition + noise
  E->>B: run prefix (optional)
  B-->>E: condition
  E->>H: bounded solver steps
  H-->>U: action chunk
```

For deadlines, replace one head call with `flow_begin` plus bounded `flow_advance` calls.

## Design rules

| Rule | Consequence |
|---|---|
| Kernel boundary | Backend ports implement one span-based interface |
| Fixed footprint | Load computes peak arena size; runtime cannot grow it |
| Hand-assembled models | No graph interpreter or dynamic dispatch in the hot path |
| External observation encoding | Core stays small and integrates with any encoder |
| Predictable latency | Optimize tail behavior and allocation count, not only throughput |
