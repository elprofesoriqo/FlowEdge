# ADR 0007: Optimize weight traffic, not the scan

Status: Accepted. Scope: `src/kernels/`, roadmap.

## Context

The plan carried from the original task list named the selective scan as the kernel to optimize with SIMD. That was an assumption, never a measurement. Running `flowedge_kernels_bench` on Mamba-130M shapes with a prefix of 4 tokens gives the actual split of a layer:

| Kernel group | Time | Share |
|---|---|---|
| `matmul` (in, out, x, dt projections) | 976 us | 84% |
| `discretize` | 103 us | 9% |
| `conv1d` and activations | 51 us | 4% |
| `selective_scan` | 29 us | 2.5% |

The two large projections also reveal which wall they sit against:

| Projection | Weights | Time | Effective bandwidth |
|---|---|---|---|
| `in_proj` | 9.44 MB | 629 us | 15.0 GB/s |
| `out_proj` | 4.72 MB | 303 us | 15.6 GB/s |
| `x_proj` | 0.49 MB | 19.9 us | 24.7 GB/s |

Two very different shapes landing on the same 15 GB/s is a DRAM signature. `x_proj` is faster only because it fits in cache. A policy runs with a short prefix, so each weight is reused a handful of times and arithmetic intensity is roughly $2\,\text{rows}$ FLOPs per byte, below the machine balance. The projections are memory-bound, not FMA-bound.

## Decision

- Optimize weight traffic first: BF16 or INT8 weights in memory, and threading for more memory parallelism.
- Do not hand-optimize `selective_scan`. At 2.5% of a layer, Amdahl caps the total gain at 2.5% even if it became free.
- If the scan path is touched, fuse `discretize` instead. It is 9%, three and a half times the scan.
- Re-measure before optimizing anything else. This table, not intuition, sets the order.

## Consequences

- Quantization moves from a precision feature to the top performance item. Halving weight bytes attacks 84% of the runtime.
- The loader's decision to widen BF16 to F32 at load is now questionable. It trades double the weight bytes for a convert-free hot path, but bytes are the bottleneck. On AVX2 the convert is a zero-extend and a shift per eight values, cheap next to a DRAM stall. Measure before changing it.
- The roadmap drops SIMD selective scan.
