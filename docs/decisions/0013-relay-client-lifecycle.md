# ADR 0013: Expose a typed SPSC client and test the process boundary

## Status

Accepted.

## Context

The Relay MVP proved each shared-memory, scheduling, worker, and trace component in one process. It
did not prove that operating-system mapping names, daemon startup, compact records, model identity,
or shutdown semantics worked together across a real process boundary. Direct ring use also required
every producer to duplicate message and validation details.

## Decision

Provide a move-only `RelayClient` in `FlowEdge::Relay`. It binds one condition producer and one action
consumer to immutable model metadata, constructs versioned requests in preallocated storage, and
validates returned action records before exposing them. Setup failures use C++23 `std::expected`;
hot-path outcomes use a small `ClientResult` value and perform no allocation.

Keep the transport SPSC. `RelayClient` is deliberately not internally synchronized, and the ring
ABI does not pretend to support multiple producers. Applications that need MPSC submission must put
an ownership gate in front of one client or allocate separate ring pairs.

Build `flowedge-relayctl` as a diagnostic producer and lifecycle tool. Add an integration test that
starts `flowedge-relayd` as an actual child process, waits for its mappings, submits a model-compatible
request, validates the action, sends the shutdown control record, and verifies a clean process exit on
Windows and POSIX hosts.

When the daemon owns the mappings, it publishes them only after checkpoint loading succeeds. A
successful client connection is therefore a readiness signal and does not consume the request's
deadline with daemon startup work.

## Consequences

- External C++ services no longer manipulate shared-memory rings or wire envelopes directly.
- Model-digest and dimension mismatches are rejected on both sides of the boundary.
- The integration gate covers process startup and teardown rather than only component simulation.
- The CLI loads a checkpoint only to obtain trustworthy metadata and generate a smoke request; it is
  not the recommended high-throughput producer.
- One client remains one SPSC endpoint, keeping atomics, ownership, and latency behavior explicit.
