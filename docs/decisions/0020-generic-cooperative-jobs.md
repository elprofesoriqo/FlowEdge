# ADR 0020: Generalize Relay with non-owning cooperative jobs and portable capsules

## Context

Relay's first worker executes one FlowEdge flow head. The scheduler concepts behind it—bounded work,
generation cancellation, model identity, and migration at safe boundaries—also apply to diffusion,
streaming SSM/LLM decode, speculative branches, and external runtimes. Encoding each workload in the
daemon protocol first would couple execution semantics to transport and make adapters depend on
FlowEdge model classes.

## Decision

Relay provides a small C++23 cooperative-job contract in `src/relay/jobs/`. `CooperativeJob` is
non-owning type erasure over a caller-owned backend and six function pointers. A descriptor gives the
scheduler a workload kind, model digest, state-schema identifier, session, generation, deadline, and
exact work-unit count. Every advance is checked against the caller's budget and the descriptor's
remaining work before progress is accepted.

The `CooperativeBackend` concept generates the function table at compile time. Named iterative,
streaming, and speculative factories assign scheduling semantics without imposing a model runtime.
Adapters define one indivisible work unit and own their tensors, output, branch policy, and mutable
state.

State migration uses a canonical little-endian envelope followed by an adapter-defined canonical
payload and a 128-bit content fingerprint. Export writes into caller-owned memory. Import validates
the complete envelope, checksum, model digest, schema, and job descriptor before calling the backend.
`StateWriter` and `StateReader` provide endian-stable integer and float encoding for payloads.

## Consequences

- Relay can host non-FlowEdge runtimes without a virtual hierarchy, heap ownership, or SDK dependency.
- Workload-specific safe points remain explicit and measurable.
- Capsule compatibility is strict; a schema change requires a new non-zero schema identifier.
- Backends must outlive their job handle and must validate a decoded payload before committing it.
- Capsule fingerprints detect accidental corruption but are not authentication primitives.
- The generic contract is currently in-process. Daemon routing and ROS 2, Zenoh, or inference-server
  adapters remain separate later integrations.
