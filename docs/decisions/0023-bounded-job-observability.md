# ADR 0023: Buffer generic job events before export

## Context

Generic worker lanes need progress, preemption, migration, and latency visibility. Direct file or
network export would add blocking work to submission and result handling.

## Decision

Emit fixed 200-byte `JobEventMessage` records into a caller-owned `JobEventBuffer`. The coordinator
drains the bounded buffer into `JobMetrics` and `TraceWriter` outside compute.

Metrics use only three fixed job-kind labels. Model, schema, and session identity remain trace fields.
Trace v2 keeps its canonical little-endian format and adds generic request, result, and event records.

## Consequences

- Pool event publication and metric recording allocate nothing after setup.
- Export backpressure cannot block a worker; new events are counted and dropped when the buffer is full.
- Migration events are defined now and emitted by the migration layer when that layer is added.
- Generic trace inspection works now; model-specific generic replay requires a production adapter.
