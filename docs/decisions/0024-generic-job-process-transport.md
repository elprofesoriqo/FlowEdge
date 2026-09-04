# ADR 0024: Reuse bounded shared memory for generic job services

## Status

Accepted.

## Context

Generic job messages, routing, worker lanes, and observability existed only behind an in-process
`JobWorkerPool`. External runtimes needed a typed endpoint and a coordinator that could preserve
results when the consumer fell behind. A second transport stack would duplicate lifecycle and
corruption rules.

## Decision

Add two move-only C++23 endpoints over the existing checksummed SPSC rings:

| Endpoint | Owns | Hot-path role |
|---|---|---|
| `JobClient` | Request producer + result consumer | Validate, submit, receive, shut down |
| `JobService` | Request consumer + result producer | Validate, admit, dispatch, publish |

`JobService` embeds a caller-owned `JobWorkerPool`; it does not own models or adapter plugins. It
publishes older results before consuming new requests. A rejected result stays in fixed service
storage, and a completed worker result stays borrowed from its lane, until the output ring accepts
it. `poll()` performs no allocation or file/network I/O.

The action-specific `RelayClient` and `flowedge-relayd` protocol remain unchanged. Generic and action
endpoints use separate ring pairs and the same versioned envelope/control record.

## Consequences

- Applications can host any frozen adapter registry behind a real process boundary.
- Output backpressure cannot silently release or overwrite a result.
- One ring pair remains strictly SPSC; MPSC ingress belongs outside the endpoint.
- The next model milestone can embed `JobService` without changing transport or worker ownership.
