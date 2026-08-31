# Roadmap

## Done

- Mamba backbone with streaming decode.
- Flow-matching head. Euler, Heun, RK4.
- CPU kernels. AVX2, NEON, scalar.
- BF16 weight storage with inline SIMD widening and adaptive threading.
- C-ABI, Python module, find_package packaging.
- Checkpoint converter.
- ULP gate against PyTorch in CI.
- Cooperative/resumable inference and portable streaming-state snapshots.
- FlowEdge Relay: typed shared-memory client, EDF admission, cancellation, portable traces, and a
  preallocated multi-worker pool.
- Shared immutable checkpoint weights and compact/spread NUMA-aware Relay worker placement.
- Worker-aware deadline admission over active and queued EDF lanes.
- Fixed-memory Relay metrics with Prometheus, JSON, and OTLP/HTTP JSON export.
- Generic allocation-free cooperative jobs, portable state capsules, and iterative, streaming, and
  speculative workload adapters.
- Generic variable-size request/result messages and frozen fixed-capacity adapter registration by
  job kind, model digest, and state schema.

## Next

- Heads: Diffusion Policy, ACT, VQ-BeT, pi0.
- Backbone: Transformer. Unlocks the transformer heads.
- Weight traffic: per-node replication experiments and INT8. See
  [ADR 0007](decisions/0007-roofline).
- Relay: connect registered generic jobs to worker pools and multi-lane admission, then add
  action-overlap policy.
- Optional ROS 2, Zenoh, and inference-server adapters over the generic job contract.
- Backends: CUDA, Tenstorrent.
- Observation encoders, which every real vision policy needs before it runs end to end. See [ADR 0006](decisions/0006-obs-encoder-out-of-scope).
