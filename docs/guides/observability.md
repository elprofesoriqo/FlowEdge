# Observability

Relay separates event capture from file/network export.

```{mermaid}
flowchart LR
  P[Worker pool] -->|fixed 200-byte events| B[Bounded buffer]
  B -->|drain outside compute| M[JobMetrics]
  B -->|drain outside compute| T[TraceWriter]
  M --> X[Prometheus / JSON / OTLP]
  T --> I[inspect / JSONL]
```

## What is recorded

| Path | Events and measurements |
|---|---|
| Action daemon | Received, accepted, rejected, published, dropped; queue/execution/end-to-end latency |
| Worker pool | Admitted, dispatched, started, preempted, migrated, terminal outcome |
| Extension events | Progress uses the same bounded event schema |
| Job identity | Kind, class, model digest, schema, session, generation, worker |
| Job work | Completed/remaining units; queue, execution, end-to-end, cancellation latency |

Job-kind labels are limited to `iterative`, `streaming`, and `speculative`. Model, session, and schema
stay in traces—not metric labels—so cardinality remains bounded.

## Record generic jobs

Provision the event buffer once and pass it to the pool:

```cpp
auto events = JobEventBuffer::create(256).value();
auto pool = JobWorkerPool::create(
    lanes, 32, costs, 64, 1, WorkerPlacement::kCompact, &events).value();
```

Drain outside submission and compute:

```cpp
JobMetrics metrics;
auto trace = TraceWriter::open("jobs.trace").value();

while (const JobEventMessage* event = events.front()) {
  metrics.record(*event);
  trace.append(*event);
  events.pop();
}
metrics.record_event_drops(events.dropped());
```

`JobEventBuffer::try_push`, metric recording, and pool event publication do not allocate. File I/O is
explicit and outside the worker path. A full buffer drops the new event and increments `dropped()`.

## Inspect a trace

```bash
./build/src/relay/flowedge-relay-trace inspect jobs.trace
./build/src/relay/flowedge-relay-trace inspect jobs.trace --jsonl
```

Trace v2 remains canonical little-endian. It now accepts condition/action, generic request/result,
and generic lifecycle records. Action replay remains unchanged.

## Export metrics

| Format | Action daemon | Generic jobs |
|---|---|---|
| Prometheus | `write_prometheus(..., RelayMetrics)` | `write_prometheus(..., JobMetrics)` |
| JSON | `write_metrics_json(...)` | `write_metrics_json(...)` |
| OTLP/HTTP JSON | `write_otlp_json(...)` | `write_otlp_json(...)` |

The action daemon can export files directly:

```bash
./build/src/relay/flowedge-relayd --model model.safetensors --create \
  --metrics-prometheus flowedge.prom \
  --metrics-json flowedge.json \
  --metrics-otlp-json flowedge-otlp.json
```

Add `--metrics-interval-ms 5000` for periodic replacement. With the default `0`, export occurs only
at graceful shutdown.

## Interpret results

| Signal | Meaning |
|---|---|
| `rejected_deadline` | Predicted work could not finish before its deadline |
| `preempted` | A newer generation requested cancellation at a safe point |
| `migration_completed` | A compatible lane restored a drained worker's capsule |
| `rejected_qos` | A lower class reached its queue reservation boundary |
| `worker_quarantines` | Lanes removed after repeated execution failures |
| `events_dropped` | The observer drained too slowly or needs a larger buffer |
| `cancellation_latency_ns` | Time from preemption request to terminal observation |
| `completed_work_units` | Runtime-neutral useful-work throughput |
