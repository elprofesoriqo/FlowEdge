# ADR 0026: Bounded worker draining

## Context

Rolling maintenance must stop new dispatch to a lane without losing accepted work or replaying a
stateful model prefix.

## Decision

- Pool creation may preallocate a fixed capsule buffer per lane; zero disables live handoff.
- `request_worker_drain` removes a lane from admission and dispatch.
- A running cooperative job exports at a work boundary and resumes on a compatible idle lane.
- The coordinator copies between fixed buffers; worker threads publish state with atomic
  acquire/release transitions and wait/notify.
- A drain is rejected when it would strand queued work. Without a target or sufficient capsule
  capacity, active work finishes on its current lane before draining.
- One execution may hand off once. The destination finishes if it is drained again.

## Consequences

Applications can rotate workers for maintenance while preserving exact model state, freshness,
results, and bounded memory. Deadline metadata moves with the job. Draining is cooperative rather
than an OS-level thread kill; latency is bounded by the backend work quantum plus capsule transfer
and restore.
