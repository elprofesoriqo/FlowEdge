# Cooperative jobs

Use one bounded contract for stateful work that may be cancelled, resumed, or moved between
preallocated worker lanes.

| Kind | Work unit | State example |
|---|---|---|
| Iterative | Denoise, ODE, planning step | Latent + solver stage |
| Streaming | Token or bounded chunk | SSM/KV state + position |
| Speculative | Draft/verify quantum | Prefix + rollback state |

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

Backend state is caller-owned and never shared concurrently. Immutable weights may be shared.
One work unit must end at a safe cancellation/migration boundary.

## Lifecycle

```{mermaid}
sequenceDiagram
  participant App
  participant Service
  participant Pool
  participant Lane
  App->>Service: submit bounded request
  Service->>Pool: identity + freshness + EDF admission
  Pool->>Lane: prepare / begin
  loop work units
    Pool->>Lane: advance(quantum)
  end
  Lane-->>Pool: result or capsule
  Pool-->>Service: ready result + events
  Service-->>App: typed result
```

## Request identity

| Field | Purpose |
|---|---|
| `kind` | Iterative, streaming, speculative |
| `model_digest` | Exact weights/configuration |
| `state_schema` | Capsule version |
| `session_id` | Stateful stream or robot |
| `generation` | Freshness/cancellation watermark |
| `deadline_ns` | Monotonic deadline; zero disables admission |
| `total_work_units` | Bounded cost and completion proof |

Relay rejects zero budgets, incompatible identity, invalid progress, early completion, and work
outside the admitted bound.

## Worker drain and migration

```cpp
pool.request_worker_drain(worker_index);
// poll JobService or ready_result() until the handoff completes
pool.resume_worker(worker_index);
```

| State at drain request | Result |
|---|---|
| Idle | Lane drains immediately |
| Running + compatible target | Capsule moves at the next boundary |
| Running + no target | Current job finishes, then lane drains |
| Accepted work would be stranded | `kWouldStrandWork`; lane stays active |

Capsules are canonical little-endian and checksum/model/schema bound. Restore validates the full
record before backend state changes.

## Production Mamba path

`MambaStreamAdapter` maps one token to one work unit. Lanes own mutable recurrence state and share
immutable weights. It preallocates token/output storage and migrates exact recurrent snapshots.

```bash
./build/mamba_relay_stream models/mamba_flow.safetensors
./build/mamba_relay_stream models/mamba_flow.safetensors
./build/flowedge_worker_drain_bench 10000
```

## Limits

| Resource | Boundary |
|---|---|
| Inline payload | 64 KiB by default; keep large tensors in model/device memory |
| Registries | Freeze before pool creation |
| Sessions | Fixed capacity; release after consuming results |
| Transport | Separate SPSC request/result rings; one service coordinator |
| Allocation | Compute, migration, scheduling, and metrics paths allocate zero after setup |
