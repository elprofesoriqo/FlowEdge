# Capabilities

## Choose by job

| Job | Entry point | Output |
|---|---|---|
| Full Mamba + flow inference | `flow_sample` | Token prefix → action chunk |
| Existing VLA/vision encoder | `external_flow_sample` / `sample_condition` | Condition → action chunk |
| Incremental decode | `mamba_forward` / `Engine.step` | Persistent recurrence |
| State transfer | `streaming_snapshot` | Exact continuation |
| Cross-process inference | `scripts/relay_demo.sh` | Deadlines, replay, metrics |
| Safe action publishing | `action_delivery_sample` | Freshness, overlap, bounds |
| Stateful custom work | `cooperative_job_sample` | Iterative / streaming / speculative |
| Managed generic service | `scripts/job_demo.sh` | QoS, recovery, trace |

## Runtime map

```{mermaid}
flowchart LR
  App[Application] --> Core[FlowEdge Core]
  App --> RC[Relay client]
  RC --> RD[Relay daemon]
  RD --> Core
  RC --> Gate[Action delivery gate]
  Gate --> Robot[Controller]
  App --> JC[Job client]
  JC --> Ring[Bounded shared memory]
  Ring --> JS[Job service]
  JS --> Pool[Worker pool]
  Pool --> Core
  Pool --> Obs[Trace + metrics]
```

## Implemented surface

| Area | Current |
|---|---|
| Models | Mamba streaming; flow head; fixed LeRobot Diffusion Policy head |
| Weights | FP32/BF16 `.safetensors`; immutable shared worker weights |
| CPU | Scalar, AVX2, NEON; adaptive workers; compact/spread placement |
| State | Versioned snapshots; model/schema-bound capsules; exact restore |
| Relay | Shared memory, EDF admission, cancellation, QoS, drain, recovery |
| Delivery | Freshness, timed overlap, bounds, rate limits |
| Observability | Portable traces; fixed-memory Prometheus/JSON/OTLP metrics |
| APIs | C ABI, CMake targets, Python |

## Contracts

| Contract | Guarantee / boundary |
|---|---|
| Allocation | Supported compute, scheduling, migration, and metrics paths allocate zero after setup |
| Concurrency | One mutable model/backend instance per lane |
| Identity | Model digest and state schema must match before restore/dispatch |
| Freshness | Older generations cannot publish as current work |
| Transport | Shared-memory rings are local SPSC endpoints |
| Deadlines | Predictive admission; not an OS hard-real-time guarantee |
| Safety | Relay is a final gate, not the robot's safety controller |

## Planned boundary

| Area | Status |
|---|---|
| Transformer + KV cache | Planned in issue #10 |
| CUDA / Metal / Vulkan / TTNN | Backend work planned |
| ONNX Runtime / TensorRT / PyTorch adapters | Planned integration layer |
| ROS 2 / Zenoh | Optional Relay adapters |
| Distributed scheduling | Deferred until local traces justify it |

FlowEdge deliberately excludes training, dataset pipelines, observation encoders, arbitrary graph
execution, and cluster scheduling.
