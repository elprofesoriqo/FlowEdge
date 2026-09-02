# Cooperative Execution

Robotics inference is a deadline problem rather than a throughput-only problem. A camera frame,
safety event, or higher-priority controller may arrive while an action head is integrating. A
single opaque inference call cannot yield, cancel stale work, or share the CPU with another
real-time task.

FlowEdge exposes a small scheduling primitive instead of embedding a scheduler:

```{mermaid}
sequenceDiagram
  participant E as External encoder
  participant R as Robot scheduler
  participant F as FlowEdge head
  E-->>R: condition vector
  R->>F: flow_begin(condition, noise, N)
  loop while budget and steps remain
    R->>F: flow_advance(step_budget, generation)
    F-->>R: current action, remaining NFE
  end
  R-->>R: publish only when remaining = 0
```

`flow_begin` performs the condition projection once and initializes fixed workspace. Each advance
executes whole Euler, Heun, or RK4 steps. It never pauses inside a matrix operation, so the kernel
layer remains simple and deterministic. The split execution is bit-identical to the corresponding
monolithic solve because it preserves operation order and solver state.

## What this solves

- A robot can interleave policy work with sensing, watchdogs, and command publication.
- A scheduler can cancel an obsolete action chunk by starting a new solve from fresher context.
- An external VLA or perception runtime can remain GPU-resident while only its compact condition
vector crosses into the CPU action head.
- A service can apply admission control in solver-step units rather than guessing from request
  counts.
- A newer generation can atomically cancel stale work. The engine checks the generation
  high-watermark between solver steps, never inside a matrix kernel or a multi-stage solver step.

The same runtime now exposes versioned Mamba decode snapshots. Their compatibility header includes
architecture, precision, dimensions, and a stable model-content digest; a payload checksum catches
truncation and corruption. Snapshot/restore is useful beyond robotics:
an SSM-based language server can fork speculative continuations, roll back rejected tokens, migrate
a session, or capture a deterministic reproducer without replaying its entire prompt.

`MambaStreamAdapter` connects that snapshot to Relay: one token is one work unit, model digest and
schema select the lane, and a state capsule resumes the stream on another compatible engine. See
[Cooperative jobs](../guides/cooperative-jobs).

## Ownership and concurrency

Workspace, recurrence state, and scratch belong to the engine. One engine supports one active solve
and one streaming session and is intentionally not internally synchronized. Applications needing
concurrency create an engine per session or put external serialization around a shared handle. This
keeps the hot path allocation-free and makes contention visible to the application scheduler.
Only the atomic cancellation high-watermark is safe to update from a concurrent scheduler thread.

See [ADR 0010](../decisions/0010-cooperative-execution) and the
[C ABI](../api/c-abi).
