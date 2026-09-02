# Capabilities

**Core** runs supported models in-process. **Relay** adds bounded scheduling, process transport,
cooperative jobs, replay, and telemetry.

## Choose a path

| Goal | Start here | Result |
|---|---|---|
| Run a Mamba + flow policy | `flow_sample` | Tokens to deterministic action chunk |
| Keep an existing VLA encoder | `external_flow_sample` | Condition vector to FlowEdge action head |
| Decode incrementally | `mamba_forward` | Persistent Mamba recurrence state |
| Move model state | `streaming_snapshot` | Exact continuation in another engine |
| Migrate a live Mamba job | `mamba_relay_stream` | Resume tokens on another Relay lane |
| Serve actions across processes | `scripts/relay_demo.sh` | Client, daemon, deadlines, replay, metrics |
| Adapt custom stateful work | `cooperative_job_sample` | Iterative, streaming, or speculative job |
| Serve custom work across processes | `routed_job_sample` | Typed result, lifecycle metrics, trace |
| Compare with PyTorch | [Performance](performance) | Matched inputs and exact commands |

## Execution map

```{mermaid}
flowchart LR
  A[Application] --> C[FlowEdge Core]
  A --> R[Relay client]
  R --> D[flowedge-relayd]
  D --> C
  A --> J[JobClient]
  J --> S[Shared-memory rings]
  S --> V[JobService]
  V --> P[JobWorkerPool]
  P --> B[Mamba or caller-owned backend]
  P --> E[Bounded event buffer]
  E --> T[Trace]
  E --> M[Metrics]
```

## Available now

| Area | Capabilities |
|---|---|
| Models | Mamba streaming; flow head with Euler, Heun, RK4 |
| Weights | FP32/BF16 `.safetensors`; shared immutable worker weights |
| CPU | Scalar, AVX2, NEON; adaptive threads; compact/spread placement |
| State | Versioned snapshots; canonical job capsules; exact restore |
| Relay | Shared memory; EDF admission; 1–8 workers; generation cancellation |
| Generic jobs | Iterative, streaming, speculative; Mamba adapter; IPC; routing; live lane drain |
| Observability | Portable traces; JSONL inspection; fixed-memory Prometheus/JSON/OTLP metrics |
| APIs | C, C++ CMake targets, Python |

## Runtime guarantees

| Contract | Expectation |
|---|---|
| Allocation | Supported compute, scheduling, migration, and metric recording allocate nothing after setup |
| Mutable state | One model/backend instance per concurrent lane |
| Identity | Model digest + state schema must match before restore or dispatch |
| Freshness | Lower generations cannot publish as current work |
| Deadlines | Admission is predictive, not an OS hard-real-time guarantee |
| Transport | Shared-memory rings are SPSC |

## Not available yet

| Area | Status |
|---|---|
| Standalone generic-job daemon | Embeddable `JobService` is available; daemon packaging is next |
| Transformer and KV cache | Planned in issue #10 |
| Diffusion Policy, DiT, π0 | Planned in issues #9 and #11 |
| CUDA, Metal, Vulkan, TTNN | Planned backends |
| ROS 2 and Zenoh | Optional Relay adapters after the production model service |
| Distributed scheduling | Deferred until local traces justify it |

FlowEdge is not an arbitrary graph runtime, training framework, or safety controller. Validate your
model and deadline on the deployment CPU.
