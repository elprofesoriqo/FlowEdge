# ADR 0012: Keep the local inference relay beside Core

## Status

Accepted.

## Context

Embodied-AI applications need freshness and physical deadlines, while generic model servers mostly
optimize aggregate throughput. Building transport and scheduling directly into Core would make the
embedded runtime depend on process lifecycle and integration concerns. Moving Relay immediately to
another repository would make its wire contracts, cancellation semantics, and cooperative execution
boundary harder to evolve atomically with Core.

## Decision

Keep Relay in `src/relay/` with a strict one-way dependency on `FlowEdge::Core`. The first milestone
is opt-in and local-only: versioned fixed-capacity messages with compact wire payloads, SPSC
shared-memory rings, a bounded EDF scheduler,
a head-only worker, generation cancellation, and checksummed replay traces. External transports and
robot-framework adapters do not enter this target.

Core remains the default build. Enabling `FLOWEDGE_RELAY` exports `FlowEdge::Relay` and builds the
`flowedge-relayd` process. Wire capacities are compile-time configuration so the shared-memory layout
is explicit and identical for every participant.

## Consequences

- Core stays independently embeddable and has no transport dependency.
- Core and Relay contracts can change in the same reviewed commit while their targets remain separate.
- The first transport is single-producer/single-consumer and host-local; multi-producer or distributed
  systems need an adapter or a later protocol milestone.
- Fixed-capacity storage keeps allocation behavior deterministic, while only the active
  condition/action prefix is copied, checksummed, and traced. Maximum capacity is chosen at build time.
- A future repository split remains possible because the dependency and exported targets are explicit.
