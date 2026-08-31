# What you can build with FlowEdge

FlowEdge has two layers. **Core** runs supported models in-process with fixed runtime memory.
**Relay** adds cross-process transport, deadline scheduling, cancellation, worker pools, traces,
metrics, and a runtime-neutral cooperative-job API.

## Choose your workflow

| I want to... | Start with | What you get |
|---|---|---|
| Run a complete Mamba + flow policy | `flow_sample` | Tokens in, deterministic action chunk out |
| Keep my existing vision/VLA encoder | `external_flow_sample` | Condition vector in, FlowEdge action head out |
| Decode a Mamba stream incrementally | `mamba_forward` | Persistent recurrence state without replaying the prefix |
| Move a streaming session to another engine | `streaming_snapshot` | Checksummed, model-bound state snapshot and exact continuation |
| Run policy inference across processes | `scripts/relay_demo.sh` | Client, daemon, deadline rejection, trace replay, and metrics |
| Scale independent requests on one host | `flowedge_relay_pool_bench` | Shared weights and 1–8 isolated worker states |
| Adapt diffusion, decode, or speculative work | `cooperative_job_sample` | Bounded steps, cancellation, and portable state capsules |
| Call the runtime from Python | `examples/flow_sample.py` | NumPy input/output over the same C++ engine |
| Port a supported checkpoint | Converter guide | Validated `.safetensors` with explicit tensor mapping |

Follow [Getting Started](getting-started) for Core or the [Relay quickstart](guides/relay-quickstart)
for a two-process deployment.

## Capabilities available today

### Model execution

- Mamba selective-SSM backbone with batch and streaming execution.
- Flow-matching action head with Euler, Heun, and RK4 solvers.
- Full backbone-plus-head checkpoints and head-only checkpoints for external encoders.
- FP32 compute and FP32/BF16 checkpoint weights.
- Scalar, AVX2, and NEON CPU kernels selected by the build/host.
- Caller-selected or bandwidth-aware automatic worker threads.
- C ABI, C++ CMake targets, and Python bindings.

After engine initialization, supported forward, streaming, and sampling paths use preplanned memory
and perform no heap allocation. One engine owns one mutable streaming/sampling state; create separate
engines for independent sessions.

### Stateful and cooperative execution

- Advance a flow solve in complete solver-step budgets.
- Cancel work by generation at a safe step boundary.
- Export/import versioned Mamba streaming snapshots.
- Adapt external iterative, streaming, or speculative C++ backends through one concept-checked API.
- Export/import canonical job capsules bound to the exact model, schema, session, generation, and
  work progress.

Generic cooperative jobs are currently an **in-process Relay API**. The existing daemon serves the
FlowEdge action-head protocol; routing arbitrary job adapters through the daemon is the next systems
milestone.

### Local inference service

- Typed `RelayClient` and `flowedge-relayd` lifecycle.
- Checksummed, variable-size SPSC shared-memory rings on Windows and POSIX.
- Bounded earliest-deadline-first scheduling.
- Freshness pruning and cooperative cancellation of older generations.
- Calibrated deadline admission over active and queued worker lanes.
- 1–8 preallocated outer workers sharing one immutable checkpoint store.
- Compact/spread NUMA-aware outer-worker placement.
- Portable condition/action traces, inspection, JSONL conversion, and deterministic replay.
- Fixed-memory metrics with Prometheus, JSON, and OTLP/HTTP JSON export.

Relay is currently a one-host service. A ring is SPSC: use one client endpoint or an application-side
MPSC gate when several producer threads share a request stream.

## What is not implemented yet

| Area | Current status |
|---|---|
| Transformer backbone and KV cache | Planned in issue #10 |
| Diffusion Policy / DiT / π0 heads | Planned in issues #9 and #11 |
| CUDA, Metal, Vulkan | Not implemented |
| Tenstorrent TTNN | Tracked in issue #14 |
| Observation/vision encoder | Intentionally outside the current engine |
| ROS 2 and Zenoh adapters | Planned as optional Relay adapters |
| Distributed scheduling | Deferred until local traces justify it |
| Generic jobs through `flowedge-relayd` | Next Relay milestone |

FlowEdge is not an ONNX-style arbitrary graph runtime, a training framework, or an OS-level hard
real-time guarantee. Use the [current performance reference](performance) and re-run the benchmarks
with your model, CPU affinity, power policy, and sustained workload before selecting a control-loop
deadline.
