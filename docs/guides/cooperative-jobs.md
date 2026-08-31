# Cooperative jobs and state migration

Relay can schedule any stateful workload that advances through bounded safe points. The runtime does
not need to be a FlowEdge model: it can be a diffusion loop, SSM decoder, transformer server,
speculative branch, planner, or accelerator session.

## Contract

A backend implements six allocation-free operations:

```cpp
struct Backend {
  bool begin() noexcept;
  fe::relay::BackendAdvance advance(std::size_t budget) noexcept;
  void cancel() noexcept;
  std::size_t state_bytes() const noexcept;
  bool save_state(std::span<std::byte>) const noexcept;
  bool load_state(std::span<const std::byte>) noexcept;
};
```

`make_iterative_job`, `make_streaming_job`, and `make_speculative_job` validate this interface with the
`CooperativeBackend` concept and return a non-owning `CooperativeJob`. The backend and every buffer it
references must outlive the handle. One handle is intentionally single-threaded; a scheduler may
request cancellation through the same serialized execution lane.

The descriptor makes scheduling and compatibility explicit:

| Field | Meaning |
|---|---|
| `kind` | Iterative, streaming, or speculative scheduling class |
| `model_digest` | Stable identity of the exact weights/configuration |
| `state_schema` | Adapter-owned non-zero payload schema identifier |
| `session_id` | Stateful conversation, robot, or stream |
| `generation` | Freshness/cancellation high-watermark |
| `deadline_ns` | Deployment clock deadline; zero may mean none |
| `total_work_units` | Exact bounded work required by this job |

One work unit must be an indivisible safe point: a complete diffusion/ODE step, token update, stream
chunk, or draft/verification quantum. `advance(budget)` rejects zero budgets, over-reported progress,
early completion, and a backend that consumes more than the remaining admitted work.

## Migration

Start and partially advance a job, export into caller-owned storage, then restore into a fresh
compatible backend:

```cpp
auto source = make_iterative_job(source_backend, descriptor).value();
source.start();
source.advance(3);

std::vector<std::byte> capsule(source.capsule_bytes());
source.export_capsule(capsule);

auto destination = make_iterative_job(destination_backend, descriptor).value();
destination.restore_capsule(capsule);
destination.advance(descriptor.total_work_units);
```

The example uses a vector only to provision storage. A deployment can reserve the maximum capsule
buffer during initialization and perform every export/import without allocating.

The envelope is canonical little-endian and covers the complete metadata and payload with a stable
128-bit content fingerprint. Import rejects corruption, truncation, unsupported versions, and any
model, schema, session, generation, deadline, kind, or work-count mismatch before invoking the
backend loader.

Adapter payloads must also be canonical. Use `StateWriter` and `StateReader` rather than copying C++
struct memory; they encode integer and IEEE-754 float bit patterns explicitly in little-endian order.
`load_state` should decode and validate into temporary values, then commit them together.

## Workload mappings

| Adapter | Work unit | Capsule payload examples |
|---|---|---|
| Iterative | One complete denoise, ODE, optimization, or planning step | latent, step index, solver stages |
| Streaming | One token, recurrence update, or bounded chunk | KV/SSM state, position, output cursor |
| Speculative | One draft or verification quantum | accepted prefix, branch cursor, rollback state |

Run `cooperative_job_sample` for migration, streaming cancellation, and speculative classification.
`flowedge_cooperative_job_bench` measures a complete partial-run/export/restore/completion cycle and
fails if the measured path allocates.

The current contract is an in-process Relay API. Shared-memory job routing, action overlap, ROS 2,
Zenoh, and inference-server adapters will build on it without adding those dependencies to Core.
