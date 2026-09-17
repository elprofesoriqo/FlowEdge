# Cooperative Execution

Robotics inference is a deadline problem rather than a throughput-only problem. A camera frame,
safety event, or higher-priority controller may arrive while an action head is integrating. A
single opaque inference call cannot yield, cancel stale work, or share the CPU with another
real-time task.

FlowEdge exposes a scheduling primitive instead of embedding a scheduler:

```{image} ../_static/figures/coop-flow.svg
:alt: flow_begin, flow_advance, publish at NFE=0
:class: fe-fig
```

`flow_begin` projects $c$ once. Each `flow_advance` runs whole Euler, Heun, or RK4 steps — never
inside a matmul. Bit-identical to the one-shot solve.

## What this solves

Yield between solver steps. Cancel a stale generation. Keep the VLM on GPU and only send $c$. Admit work in NFE units. Mamba snapshots restore without replaying the prefix. [Cooperative jobs](../guides/cooperative-jobs).

## Ownership and concurrency

Workspace, recurrence state, and scratch belong to the engine. One engine supports one active solve
and one streaming session and is intentionally not internally synchronized. Applications needing
concurrency create an engine per session or put external serialization around a shared handle. This
keeps the hot path allocation-free and makes contention visible to the application scheduler.
Only the atomic cancellation high-watermark is safe to update from a concurrent scheduler thread.

See [ADR 0010](../decisions/0010-cooperative-execution) and the
[C ABI](../api/c-abi).
