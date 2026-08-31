# ADR 0021: Add validated generic messages and frozen adapter registration

## Context

The cooperative-job API can execute and migrate caller-owned backends, but it previously had no
transport identity or bounded way to select an adapter. Adding model-specific branches to the daemon
would recreate a closed worker hierarchy and make optional runtimes compile-time Relay dependencies.
Allowing arbitrary factories on the request path would also introduce ownership ambiguity and
unbounded initialization work.

## Decision

Relay adds `kJobRequest` and `kJobResult` after the existing condition, action, and shutdown message
IDs. Their versioned metadata carries a `JobDescriptor`, timestamp, exact payload length, progress,
and a typed result code. Payload capacity is fixed at build time by
`FLOWEDGE_RELAY_MAX_JOB_PAYLOAD_BYTES`, while `wire_size` sends only the used prefix. Construction
initializes headers and metadata without clearing the unused payload tail.

`JobAdapterRegistry` uses caller-provided entry storage. Initialization registers one caller-owned
backend instance under `(job kind, model digest, state schema)`, rejects duplicate or oversized
entries, and then freezes the table. Serving performs a linear bounded lookup, validates the adapter's
payload limit, prepares the backend, and returns a `RoutedJob` over the existing cooperative contract.
The `RoutedBackend` C++23 concept generates prepare, execution, migration, cancellation, and result
bridges without virtual ownership.

One registration represents one mutable execution lane. Concurrent lanes provision independent
backend instances and registries (or distinct entries) while immutable weights may remain shared.
Freezing must happen before publishing the registry to worker threads.

## Consequences

- Existing v1 flow message IDs and layouts remain unchanged.
- Generic requests/results can already use the checksummed variable-size shared-memory ring.
- Route lookup, binding, execution, and result creation allocate no heap memory after setup.
- Large tensors should normally remain in backend-owned or out-of-band memory; increasing the fixed
  inline payload also increases the maximum ring-slot footprint.
- The current daemon does not consume generic message kinds yet. Pool dispatch, per-kind admission,
  traces, and metrics remain the next integration step.
