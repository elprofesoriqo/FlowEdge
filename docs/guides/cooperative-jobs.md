# Cooperative jobs

Use one Relay contract for bounded stateful work.

| Kind | One work unit | Typical state |
|---|---|---|
| Iterative | Denoise, ODE, optimization, planning step | Latent, solver stage, step |
| Streaming | Token, recurrence update, bounded chunk | SSM/KV state, position |
| Speculative | Draft or verification quantum | Accepted prefix, branch, rollback state |

## Backend contract

```cpp
struct Backend {
  bool prepare(std::span<const std::byte>) noexcept;
  bool begin() noexcept;
  fe::relay::BackendAdvance advance(std::size_t budget) noexcept;
  void cancel() noexcept;
  std::size_t state_bytes() const noexcept;
  bool save_state(std::span<std::byte>) const noexcept;
  bool load_state(std::span<const std::byte>) noexcept;
  std::span<const std::byte> result() const noexcept;
};
```

`RoutedBackend` checks this surface at compile time. The backend remains caller-owned and must outlive
its registry and worker pool.

## Identity and scheduling

| Descriptor field | Contract |
|---|---|
| `kind` | Iterative, streaming, or speculative cost class |
| `model_digest` | Exact weights/configuration identity |
| `state_schema` | Adapter payload version |
| `session_id` | Stateful stream or robot |
| `generation` | Freshness and cancellation watermark |
| `deadline_ns` | Monotonic deadline; zero disables admission |
| `total_work_units` | Exact bounded work required |

One work unit must end at a safe cancellation/migration point. Relay rejects zero budgets,
over-reported progress, early completion, incompatible state, and work beyond the admitted bound.

## Route and observe

```{mermaid}
sequenceDiagram
  participant App
  participant Client as JobClient
  participant Service as JobService
  participant Pool as JobWorkerPool
  participant Lane as Frozen adapter lane
  participant Events as JobEventBuffer
  App->>Client: try_submit(request)
  Client->>Service: checksummed shared-memory ring
  Service->>Pool: submit(request, now)
  Pool->>Pool: identity + freshness + EDF admission
  Pool->>Lane: prepare(payload)
  loop bounded work
    Pool->>Lane: advance(quantum)
  end
  Lane-->>Pool: JobResultMessage
  Pool-->>Events: lifecycle + timing
  Pool-->>Service: ready_result()
  Service-->>Client: result ring
  Client-->>App: try_receive(result)
```

Provision one backend and frozen registry per lane:

```cpp
std::array<Backend, 2> backends;
std::array<std::array<JobAdapterRegistration, 1>, 2> entries;
std::array<JobAdapterRegistry, 2> lanes{
    JobAdapterRegistry{entries[0]}, JobAdapterRegistry{entries[1]}};
for (std::size_t i = 0; i < lanes.size(); ++i) {
  lanes[i].add(make_routed_adapter(
      backends[i], job_route(descriptor), max_input_bytes, max_output_bytes));
  lanes[i].freeze();
}
auto events = JobEventBuffer::create(256).value();

auto pool = JobWorkerPool::create(
    lanes, 32,
    JobCostPolicy{.iterative_ns = step_ns,
                  .streaming_ns = token_ns,
                  .speculative_ns = quantum_ns},
    64, 1, WorkerPlacement::kCompact, &events).value();

pool.submit(request, now_ns);
while (pool.ready_result() == nullptr)
  std::this_thread::yield();
consume(*pool.ready_result());
pool.release_ready_result();
pool.release_session(descriptor.session_id);
```

For another process, connect the same pool to a typed service:

```cpp
auto service = JobService::create("jobs-in", "jobs-out", pool, 32).value();
while (!service.stopped()) {
  if (service.poll() == JobServiceResult::kIdle)
    std::this_thread::yield();
}
```

The producer uses `JobClient::connect`, `try_submit`, and `try_receive`. Full output rings retain the
result inside the service or worker lane. Drain `events` into `JobMetrics` or `TraceWriter` outside
compute. See [Observability](observability).

## Move a running job

```cpp
auto source = make_iterative_job(source_backend, descriptor).value();
source.start();
source.advance(3);
source.export_capsule(preallocated_capsule);

auto destination = make_iterative_job(destination_backend, descriptor).value();
destination.restore_capsule(preallocated_capsule);
destination.advance(descriptor.total_work_units);
```

Capsules are canonical little-endian and fingerprint the full record. Restore validates version,
model, schema, session, generation, deadline, kind, work count, and checksum before backend state is
committed. Use `StateWriter`/`StateReader` for adapter payloads.

## Limits

| Limit | Behavior |
|---|---|
| Inline payload | 64 KiB by default; keep large tensors in runtime/device memory |
| Registries | Frozen before pool creation; lookup is `(kind, model, schema)` |
| Sessions | Fixed capacity; call `release_session` after consuming results |
| Mutable backend | Never shared concurrently; immutable weights may be shared |
| Generic transport | Separate SPSC request/result rings; `JobService::poll()` is single-coordinator |

## Run

```bash
./build/cooperative_job_sample
./build/routed_job_sample jobs.trace
./build/src/relay/flowedge-relay-trace inspect jobs.trace --jsonl
./build/flowedge_cooperative_job_bench 1000000
```

The benchmark covers migration, direct and pooled routing, process-compatible transport, lifecycle
metrics, and runtime allocation checks.
