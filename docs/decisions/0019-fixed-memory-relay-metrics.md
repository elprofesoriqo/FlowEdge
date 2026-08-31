# ADR 0019: Record Relay metrics in fixed memory and export off the compute path

## Context

Daemon shutdown counters were useful during development but could not feed production dashboards or
correlate deadline outcomes with queue and execution latency. Pulling a telemetry SDK into Core or
allocating labels for every request would violate the dependency and jitter constraints.

## Decision

Relay owns a fixed-size `RelayMetrics` registry. It records typed outcomes, corrupt/drop counts,
queue and busy-worker high-water marks, worker failures, and 65-bucket power-of-two histograms for
queue, execution, and end-to-end nanoseconds. Recording performs only bounded integer/array updates.

The daemon can write three optional formats: Prometheus text, a compact JSON snapshot, and a valid
OTLP/HTTP JSON request body. Export is at graceful shutdown by default. An explicit bounded
`--metrics-interval-ms` enables periodic replacement of the same files; this I/O occurs in the daemon
loop but never inside a Core engine or worker compute thread.

## Consequences

- Production integrations need no telemetry dependency in Core or Relay.
- Prometheus node-exporter textfile collection and OpenTelemetry Collector ingestion are possible
  without changing inference code.
- Histograms have bounded precision and cardinality by design; arbitrary runtime labels are excluded.
- Periodic file output is opt-in because filesystem stalls can affect transport-loop latency. A team
  requiring isolated live export can read snapshots from a separate adapter in a later integration.
