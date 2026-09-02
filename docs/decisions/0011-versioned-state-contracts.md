# ADR 0011: Versioned State and Request Contracts

## Status

Accepted

## Context

Raw recurrence-state copies and untyped condition vectors are adequate inside one process, but not
for session migration, shared-memory workers, deterministic replay, or stale-request cancellation.
A dimension-compatible state from different weights can be numerically valid while being
semantically wrong, and partial writes must not silently enter a live controller.

## Decision

- Give every loaded checkpoint a stable 128-bit content fingerprint and explicit architecture,
  precision, and dimension identity.
- Serialize Mamba state into a fixed 96-byte little-endian version-1 envelope followed by the raw
  state payload. Validate magic, version, identity, exact size, and checksum before copying.
- Keep all snapshot storage caller-owned; serialization and restoration allocate nothing.
- Define fixed-width C metadata for conditions and actions containing source timestamp, deadline,
  cancellation generation, model digest, dimensions, solver, and remaining NFE.
- Maintain an atomic newest-generation high-watermark. Cooperative integration checks it between
  complete steps so cancellation cannot change numerical operation order inside a step.

The fingerprint is a compatibility and replay key, not an authentication primitive. Untrusted
artifacts still require a cryptographic signature at a package or transport boundary.

## Consequences

- Wrong-model, wrong-precision, truncated, and corrupt snapshots fail before mutable state changes.
- Relay and external schedulers can use one Core-owned schema without Core depending on transport.
- Existing snapshot blobs are intentionally incompatible because raw state had no safe discriminator.
- Checkpoint load performs one fast content-hash pass; inference and snapshot hot paths remain
  allocation-free.
