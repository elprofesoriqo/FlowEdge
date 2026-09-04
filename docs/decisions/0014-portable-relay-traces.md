# ADR 0014: Write canonical portable Relay traces

## Status

Accepted.

## Context

Relay trace v1 wrote C++ structs directly. It was fast and checksummed, but its integer byte order,
enum representation, padding, and floating-point payload representation were implicit host ABI
details. That made a trace unsuitable as a durable artifact for comparing Windows and Linux runs,
sharing failures between companies, or replaying an edge capture on a development workstation.

## Decision

Write trace v2 with explicit field-by-field little-endian encoding. Integers use fixed widths, floats
store their IEEE-754 bit pattern as a little-endian `uint32`, and model digests remain exact bytes.
Both the 16-byte file header and every 32-byte record header are serialized without native structs.
The checksum covers the canonical encoded message, not host memory.

Decode into the existing native `ConditionMessage`, `ActionMessage`, or `ControlMessage` only after
the record header, bounds, checksum, envelope, dimensions, and message semantics pass validation.
Continue reading v1 files on little-endian hosts so development traces from the MVP remain useful;
v1 is rejected on big-endian systems because its original ABI cannot be inferred safely.

Build `flowedge-relay-trace` with two offline workflows:

- `inspect` emits a human summary or newline-delimited JSON records;
- `replay` reruns every captured condition through a compatible checkpoint and compares completed
  action vectors with a configurable absolute tolerance.

## Consequences

- New traces are portable across supported Windows and POSIX hosts and independent of C++ padding.
- Checksums detect corruption in the portable representation before it reaches the model runtime.
- The live shared-memory ring remains a same-host native ABI; portability is a property of durable
  traces, not a claim that heterogeneous machines can share one mapping.
- Replay reports non-complete actions separately because an isolated offline solve cannot reproduce
  cancellation timing that was not captured as step-level events.
- Adding protocol fields requires an explicit trace codec update and a new format version when the
  canonical schema changes incompatibly.
