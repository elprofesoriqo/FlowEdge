---
description: FlowEdge repository rules for coding agents.
---

# FlowEdge agent rules

[`.agents/engineering-standard.md`](../engineering-standard.md) is authoritative. Apply it to every
C++ change and run [`checkcorrect.md`](checkcorrect.md) before committing.

The current dependency direction is:

```text
applications -> FlowEdge::Relay -> FlowEdge::Core
                              Core -/-> Relay
```

Core lives in `src/core/`; Relay lives in `src/relay/`. Do not use the retired top-level
`src/kernels`, `src/heads`, `src/arena`, or `src/loader` paths.
