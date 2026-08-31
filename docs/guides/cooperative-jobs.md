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

Provision adapter entries and mutable backends during initialization, then freeze before worker
threads can see the registry:

```cpp
Backend backend;
std::array<JobAdapterRegistration, 4> entries{};
JobAdapterRegistry registry{entries};

registry.add(make_routed_adapter(
    backend, job_route(descriptor), max_input_bytes, max_output_bytes));
registry.freeze();

JobRequestMessage request;
make_job_request(request, sequence, descriptor, now_ns, encoded_input);
auto job = registry.bind(request).value();
job.start();
job.advance(descriptor.total_work_units);

JobResultMessage result;
job.write_result(result, finished_ns);
```

The lookup key is the exact `(kind, model_digest, state_schema)` tuple. Unknown routes, duplicate
registration, malformed messages, and adapter-specific payload overflows are typed failures. One
registration owns one mutable backend lane; provision a backend/registry per concurrent lane while
sharing immutable model weights outside it.

```{mermaid}
sequenceDiagram
  participant Client
  participant Message as JobRequestMessage
  participant Registry as Frozen registry
  participant Backend as Adapter backend
  Client->>Message: make_job_request(descriptor, payload)
  Client->>Registry: bind(validated message)
  Registry->>Backend: prepare(payload)
  Registry-->>Client: RoutedJob
  loop bounded safe points
    Client->>Backend: advance(work budget)
    Backend-->>Client: checked progress
  end
  Client->>Backend: result()
  Client->>Message: write typed JobResultMessage
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
Run `routed_job_sample` for request creation, frozen-registry lookup, execution, and typed result
encoding. `flowedge_cooperative_job_bench` measures both complete lifecycles and fails if either path
allocates.

Generic messages can traverse an appropriately sized `SharedMemoryRing`, but `flowedge-relayd` does
not consume them yet. Worker-pool dispatch, action overlap, ROS 2, Zenoh, and inference-server
adapters will build on this foundation without adding those dependencies to Core.
