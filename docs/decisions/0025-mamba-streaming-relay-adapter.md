# ADR 0025: Mamba streaming Relay adapter

## Context

The generic Relay contract was proven with synthetic backends. A production model must preserve
Core output, identity, bounded memory, cancellation points, and recurrent state across lanes.

## Decision

- `MambaStreamAdapter` maps one token to one `JobKind::kStreaming` work unit.
- Routes use the Core model digest plus the versioned `mamba-v1` state schema.
- One adapter owns one mutable engine; lanes may share a loaded `fe_weights` store.
- Setup fixes maximum tokens and allocates request/output storage once.
- Capsules contain tokens, progress, hidden output, and the exact Core decode snapshot in canonical
  little-endian form.
- Setup uses `std::expected`; buffers use `std::span`; `RoutedBackend` verifies the adapter contract.

## Consequences

Mamba streams can run, cancel, and resume through the generic worker pool without replaying the
prefix or allocating in the hot path. A capsule is model/schema/session/generation specific and may
be larger than the recurrent snapshot because it also carries the request and current output.

`flowedge-relayd` remains the action-head daemon. Generic Mamba serving is embedded with
`JobService` until the draining-aware generic daemon is implemented.
