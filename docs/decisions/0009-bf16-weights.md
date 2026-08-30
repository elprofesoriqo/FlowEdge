# ADR 0009: BF16 Weight Storage and Inline Widening

## Status
Accepted

## Context
Profiling shows matrix multiplications (`in_proj`, `out_proj`) are memory-bandwidth bound, not compute-bound. Widening `BF16` weights to `F32` at load time doubles the memory footprint and halves DRAM bandwidth efficiency.

## Decision
Store projection weights as `BF16` in the arena. Widen to `F32` dynamically in the innermost hardware loops.

- **AVX2:** Load 8 `BF16` elements (`_mm_loadu_si128`), then zero-extend and shift directly into an `F32` register (`_mm256_slli_epi32`).
- **NEON:** Inline widening using `vshlq_n_u32` and `vreinterpretq_f32_u32`.
- **Exclusions:** 1D convolutions, biases, and small vectors stay `F32` to skip widening overhead on non-bottlenecked paths.

## Consequences
- **Positive:** Bandwidth utilization is doubled. Large matmuls achieve ~1.7x speedup. Arena slab size drops significantly.
- **Negative:** Precision is limited to `BF16` (truncates mantissa but retains dynamic range). Inner kernels are slightly more complex.
