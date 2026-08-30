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

## Next

- Heads: Diffusion Policy, ACT, VQ-BeT, pi0.
- Backbone: Transformer. Unlocks the transformer heads.
- Weight traffic: per-node replication experiments and INT8. See
  [ADR 0007](decisions/0007-roofline).
- Relay: production metrics, action overlap, and portable state migration.
- Cooperative adapters for iterative diffusion, streaming SSM/LLM decode, and speculative branches.
- Backends: CUDA, Tenstorrent.
- Observation encoders, which every real vision policy needs before it runs end to end. See [ADR 0006](decisions/0006-obs-encoder-out-of-scope).
