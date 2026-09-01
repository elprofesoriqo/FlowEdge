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

## Route requests to an adapter

A routable backend adds two allocation-free operations to the cooperative contract:

```cpp
bool prepare(std::span<const std::byte> request) noexcept;
std::span<const std::byte> result() const noexcept;
```

`prepare` validates and binds adapter-specific input before `begin`; `result` exposes backend-owned
output for copying into a typed result message. The C++23 `RoutedBackend` concept checks the complete
eight-operation surface.

Provision one mutable backend and frozen registry per concurrent lane:

```cpp
std::array<Backend, 2> backends;
std::array<JobAdapterRegistration, 4> first_entries{};
std::array<JobAdapterRegistration, 4> second_entries{};
std::array<JobAdapterRegistry, 2> lanes{
    JobAdapterRegistry{first_entries}, JobAdapterRegistry{second_entries}};

for (std::size_t lane = 0; lane < lanes.size(); ++lane) {
  lanes[lane].add(make_routed_adapter(
      backends[lane], job_route(descriptor), max_input_bytes, max_output_bytes));
  lanes[lane].freeze();
}

auto pool = JobWorkerPool::create(
    lanes, 32,
    JobCostPolicy{.iterative_ns = measured_step_ns,
                  .streaming_ns = measured_token_ns,
                  .speculative_ns = measured_quantum_ns});

JobRequestMessage request;
make_job_request(request, sequence, descriptor, now_ns, encoded_input);
pool->submit(request, now_ns);

const JobResultMessage* result = nullptr;
while ((result = pool->ready_result()) == nullptr)
  std::this_thread::yield();
consume(*result);
pool->release_ready_result();
pool->release_session(descriptor.session_id);
```

The lookup key is the exact `(kind, model_digest, state_schema)` tuple. Unknown routes, stale
generations, unreachable deadlines, capacity limits, malformed messages, and payload overflows are
typed results. Registries and backends must outlive the pool. Share immutable model weights outside
the lanes. `max_sessions` bounds retained freshness state; release a finished session explicitly.

```{mermaid}
sequenceDiagram
  participant Client
  participant Message as JobRequestMessage
  participant Pool as JobWorkerPool
  participant Lane as Frozen adapter lane
  Client->>Message: make_job_request(descriptor, payload)
  Client->>Pool: submit(message, now)
  Pool->>Pool: freshness + EDF admission
  Pool->>Lane: prepare(payload)
  loop bounded safe points
    Pool->>Lane: advance(work quantum)
    Lane-->>Pool: checked progress
  end
  Lane-->>Pool: typed JobResultMessage
  Pool-->>Client: ready_result()
```

Generic messages default to a 64 KiB inline payload, configurable with
`FLOWEDGE_RELAY_MAX_JOB_PAYLOAD_BYTES`. Only the used prefix is initialized, checksummed, and moved
through a ring. Keep large tensors in adapter-owned/device memory and pass a validated handle when
possible; raising the inline limit increases each maximum ring slot.

## Workload mappings

| Adapter | Work unit | Capsule payload examples |
|---|---|---|
| Iterative | One complete denoise, ODE, optimization, or planning step | latent, step index, solver stages |
| Streaming | One token, recurrence update, or bounded chunk | KV/SSM state, position, output cursor |
| Speculative | One draft or verification quantum | accepted prefix, branch cursor, rollback state |

Run `cooperative_job_sample` for migration, streaming cancellation, and speculative classification.
Run `routed_job_sample` for queued worker execution and typed results.
`flowedge_cooperative_job_bench` measures direct and pooled routing and fails on a runtime allocation.

Generic messages can traverse an appropriately sized `SharedMemoryRing`, but `flowedge-relayd` does
not consume them yet. Transport integration, action overlap, ROS 2, Zenoh, and inference-server
adapters build on this pool without adding those dependencies to Core.
