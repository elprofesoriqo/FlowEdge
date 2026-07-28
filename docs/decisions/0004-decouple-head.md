# ADR 0004: Decouple the head from the backbone

Status: Accepted. Scope: `src/heads/`, `src/models/`.

## Context

The flow head first lived inside `src/models/mamba/`. A head only needs a conditioning vector, not a specific backbone. Coupling blocks the backbone-by-head matrix.

## Decision

- Heads live in `src/heads/`. Backbones live in `src/models/`.
- A head consumes a conditioning vector and returns an action. It knows nothing about the backbone.
- The contract is a float vector of size `d_model`. Any backbone that produces it can drive any head.

## Consequences

- A new head is a self-contained module. A new backbone is too.
- Transformer with flow, Mamba with diffusion, and other pairs become real combinations, not rewrites.
