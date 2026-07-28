# ADR 0005: Hand-assembled architectures, not a graph runtime

Status: Accepted. Scope: whole project.

## Context

FlowEdge could ingest an arbitrary model graph, like ONNX Runtime. Or it could hand-write each architecture, like llama.cpp.

## Decision

- Each architecture is hand assembled from shared kernels. There is no graph interpreter.
- Reuse lives in the primitives: kernels, arena, loader, converter, and the head and backbone contracts.

## Consequences

- Zero hot-path allocation, fixed memory, hard real-time, and a tiny binary are possible because the architecture is known statically. A graph interpreter would give that up.
- The cost is manual work per architecture. The first model of a family pays for its kernels. The rest reuse them.
- To run any model automatically, use a graph runtime. FlowEdge is not that.
