# ADR 0002: Mamba block and zero-alloc forward

Status: Accepted. Scope: `src/core/models/mamba/`.

## Context

Level II wires the kernels and arena weights into a real Mamba forward. Hard rule: no allocation on the runtime path.

## Decision

- Resolve weights once at construction. The constructor derives `MambaConfig` from tensor shapes and stores raw pointers. No lookups in the hot path.
- `forward` allocates nothing. Each layer carves scratch from the arena and rewinds the cursor per layer. A 24-layer run reuses one small region.
- Fixed `std::array<Layer, kMaxLayers>`. No dynamic containers.
- Config is derived from the checkpoint. A Mamba `.safetensors` is self-describing.
- Discretization and scan are fused behind `discretize_and_scan` to avoid a
  separate memory pass while preserving the state-major access pattern.

Per-layer pipeline: RMSNorm, in_proj, conv1d, SiLU, x_proj, dt_proj, softplus,
`discretize_and_scan`, gate, out_proj, residual. Then a final RMSNorm.

## Consequences

- Runs the real Mamba-130M forward, checked against PyTorch by the ULP gate. The observed error on the reference checkpoint is around 1e-6, well inside the gate's `2e-3` relative bound.
- Streaming decode reuses the same block with a persistent conv window and SSM state. It matches the batch forward.
- The fused scan is now part of the tested CPU path; future work should focus on
  weight traffic and backend ports before more scan tuning.
