# ADR 0022: Route generic jobs through caller-provisioned worker lanes

## Context

The generic registry binds one backend synchronously. Production adapters need bounded queueing,
parallel lanes, freshness cancellation, and deadline admission without moving runtime dependencies
into Core.

## Decision

Add `JobWorkerPool`. Each of its 1–8 lanes receives one caller-owned frozen registry and mutable
backend instance. The pool owns a fixed EDF queue, worker threads, session-generation watermarks,
typed results, and separate iterative, streaming, and speculative work-unit costs.

All storage is allocated during `create`. Execution advances by a bounded work quantum so a newer
generation can cancel at an adapter-defined safe point. Immutable model weights remain shareable
outside the lanes.

## Consequences

- Generic runtimes use Relay scheduling without becoming Relay or Core dependencies.
- One mutable backend cannot be shared concurrently; callers provision one instance per lane.
- Session watermarks use fixed storage and are reclaimed explicitly after all results are consumed.
- Deadline admission is disabled for an uncalibrated job kind.
- The current daemon remains specific to FlowEdge condition/action messages; embedders own generic
  adapter registration and transport integration.
- Optional lifecycle events flow into a caller-owned bounded buffer; ADR 0023 defines export.
