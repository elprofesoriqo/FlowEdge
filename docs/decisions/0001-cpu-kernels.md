# ADR 0001: CPU kernel strategy

Status: Accepted. Scope: `src/kernels/cpu/`.

## Context

Level III kernels for the forward pass. Targets AVX2 on x86 edge and NEON on Jetson. Parity with PyTorch is required, so fast-math is off. Kernels are allocation-free and run on arena buffers.

## Decision

- Explicit intrinsics for the seven kernels that carry the runtime: `matmul`, `silu`, `softplus`, `gate_silu`, `discretize`, `selective_scan` and `scan_step`.
- `exp` and `log` use hand-written Cephes polynomials, about 1 ULP. The compiler folds a scalar `exp` into a per-element libm call, so intrinsics are the only way to vectorize it. Constants are shared between the ISA files in `cephes.h`.
- `conv1d_causal`, `conv1d_step` and `rmsnorm` stay scalar and live once in `kernels_common.cc`. They are under 3% of a layer, too little to justify three ISA copies.
- Scan layout is state-major `[t][n][c]` so the inner channel loop is contiguous.

## Consequences

- Portable. A scalar fallback compiles where SIMD is absent.
- `matmul` is the dominant cost and the first optimization target.
- The `exp` approximation is a bounded, deliberate deviation. Full parity is enforced by the ULP gate.
