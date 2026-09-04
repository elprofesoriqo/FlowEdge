# ADR 0027: Gate action chunks at the controller boundary

## Context

Inference produces flattened action chunks slower than many control loops consume them. Replacing a
chunk abruptly can introduce discontinuities; accepting stale, mismatched, or unsafe output can move
the robot from invalid context.

## Decision

- Interpret a complete `ActionMessage` as a versioned, zero-copy `ActionChunkView`.
- Bind delivery to model digest and session; require increasing generation and sequence.
- Select steps by monotonic control time and linearly blend a bounded replacement prefix.
- Apply per-axis absolute and delta limits with explicit reject or clamp policy.
- Copy only configured limits and active chunk values into fixed-capacity storage.
- Keep the existing action wire protocol unchanged.

## Consequences

Policy and controller rates are decoupled without a growing queue. Replacement, publication, and
safety checks allocate nothing. The gate detects bad software output but does not replace physical
limits, watchdogs, or certified safety systems.
