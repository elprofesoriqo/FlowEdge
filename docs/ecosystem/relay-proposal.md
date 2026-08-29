# FlowEdge Relay: In-Repository Component Proposal

FlowEdge Core remains the small, allocation-free execution library. An optional component in this
repository, provisionally **FlowEdge Relay**, should solve the system problem around it:
freshness-aware, deadline-aware inference between perception/VLA services and robot controllers.

## Problem

Companies assembling embodied-AI systems repeatedly build the same fragile layer: shared-memory or
network transport, action-chunk queues, stale-request cancellation, model-session state, deadline
admission, and replay traces. Generic model servers optimize request throughput; robotics needs the
newest valid action before a physical deadline. LLM infrastructure has a related need for resumable
state, preemption, and fair scheduling of long-lived sessions.

## Proposed shape

```{mermaid}
graph LR
  S[Sensor / prompt] --> E[Encoder or VLA service]
  E -->|condition + timestamp| R[FlowEdge Relay]
  R -->|shared memory| H[FlowEdge head workers]
  H -->|candidate chunk| R
  R --> C[Freshness and deadline gate]
  C --> A[Robot adapter]
  R --> X[Replay capsules and metrics]
```

Relay would provide:

- a versioned condition/action protocol with timestamps, schema hashes, model digest, solver budget,
  and cancellation generation;
- local shared-memory rings first, with optional ROS 2 and gRPC adapters kept in separate packages;
- earliest-deadline-first admission measured in solver steps or calibrated NFE cost;
- action-chunk freshness rules, overlap handling, and explicit degraded/fallback actions;
- Mamba/SSM state capsules for session migration, speculative branches, and deterministic replay;
- latency histograms and deadline-miss events without placing telemetry in the inference kernels;
- adapter SDKs for PyTorch, JAX, TensorRT, ONNX Runtime, and common VLA serving processes.

## Repository boundary

Transport libraries, ROS distributions, authentication, telemetry, and daemon lifecycles have a
different dependency footprint from a static C++ inference engine. The repository therefore keeps
the boundary explicit instead of putting these concerns into the engine:

- `src/core/` builds `FlowEdge::Core`, owns inference state and kernels, and has no Relay dependency;
- `src/relay/` will build `FlowEdge::Relay` and the `flowedge-relayd` executable when implementation
  begins;
- transport and robot integrations remain optional Relay adapters;
- Relay may be packaged and versioned independently even while both components share one repository.

Keeping Core and Relay together lets state capsules, protocols, model metadata, and cooperative
execution evolve atomically. The one-way dependency preserves Core's embeddability. A repository
split remains possible later if the release cadence or contributor community genuinely diverges.
No empty `src/relay/` placeholder is kept before the first implementation milestone.

The first milestone should stay narrow: one-machine shared memory, one encoder producer, a pool of
head-only FlowEdge engines, cancellation by generation number, and a replay log. Distributed
scheduling should follow only after real traces show that local scheduling is insufficient.

## Reuse outside robotics

The protocol is deliberately condition-vector and state-capsule oriented rather than robot-message
specific. ML systems teams could use the same worker and scheduler for latent decoders, speculative
SSM/LLM branches, iterative generative heads, or any model stage that exposes bounded cooperative
steps. Robot adapters translate actions and safety semantics at the edge of the system.
