# Kernels

One header declares every kernel, and each backend implements it.

```{mermaid}
%%{init: {'theme':'base','flowchart':{'htmlLabels':false,'nodeSpacing':28,'rankSpacing':34,'useMaxWidth':false},'themeVariables':{'primaryColor':'#f6ead0','primaryBorderColor':'#7b2733','lineColor':'#7b2733','primaryTextColor':'#2b2521','secondaryColor':'#eaddbf','tertiaryColor':'#faf3e2','fontFamily':'system-ui, -apple-system, Segoe UI, Roboto, sans-serif','fontSize':'13px'}}}%%
graph TD
  I["kernels.h"]
  I --> CPU["cpu: avx2 / neon / scalar"]
  I -.-> CU["CUDA (design target)"]
  I -.-> TT["Tenstorrent (design target)"]
```

The production backend is picked at build time:

```bash
cmake -B build -DFLOWEDGE_BACKEND=cpu
```

`cpu` selects AVX2, NEON, or scalar code for the host. `avx512` is an explicit x86 build option.
`scalar` explicitly selects the portable implementation for compatibility builds, differential
testing, sanitizers, and CPUs where an AVX2 deployment baseline is unsuitable.
CUDA, Metal, Vulkan, and Tenstorrent files are placeholders for future ports and are rejected by
CMake until they implement the complete kernel interface.

The CPU backend splits by ISA. `kernels_avx2.cc`, `kernels_neon.cc`, and
`kernels_scalar.cc` all implement the same public surface, and CMake picks the
best file for the host. Every kernel takes `std::span`, owns no memory, and
never allocates over 64-byte-aligned caller buffers.

The set is `matmul` for F32 and BF16 weights, `conv1d_causal`,
`conv1d_step`, `rmsnorm`, `silu`, `softplus`, `gate_silu`, and
`discretize_and_scan`.

## The exp and log kernels

`exp` and `log` use the Cephes method, because the compiler folds a scalar `exp` back into a per-element libm call. `exp` reduces its argument to a small interval and then evaluates a polynomial:

$$
n = \left\lfloor x \log_2 e \right\rceil, \qquad r = x - n \ln 2, \qquad \exp(x) = 2^{n}\, P(r)
$$

$P$ is a degree-5 minimax polynomial on $r \in [-\tfrac{\ln 2}{2}, \tfrac{\ln 2}{2}]$, and $2^{n}$ is applied by writing $n$ into the float exponent. The result is within about 1 ULP. The vector forms are `exp8` on AVX2 and `exp4` on NEON.

## Why this way

Almost everything here is bandwidth-bound. That sets the optimization order.

`matmul` dominates the runtime, about 84 percent of a layer. A square GEMM would be compute-bound, with $O(N^3)$ FLOPs over $O(N^2)$ bytes, but that is not the regime a policy runs in. The prefix is short, so each weight is read only a handful of times and the arithmetic intensity falls to roughly $2\,\text{rows}$ FLOPs per byte, well under the machine balance. Measurement agrees: `in_proj` and `out_proj` have very different shapes and both land near 15 GB/s, which is a DRAM ceiling rather than an FMA one, while the small cache-resident `x_proj` runs half again faster. So the lever on `matmul` is weight traffic, not FMA blocking. Storing weights in BF16 halves the bytes, INT8 quarters them, and threading buys more memory parallelism. See [ADR 0007](../decisions/0007-roofline).

The scan path is bandwidth-bound too, so the CPU backend fuses discretization and
state update into `discretize_and_scan`. It streams $\bar{A}$ and $\bar{B}u$
once with almost no reuse, so intensity is near one FLOP per byte and the ALU
waits on memory. Fusing removes an intermediate pass and keeps the model code
small.

`matmul`, the activations, and the fused scan carry explicit intrinsics on SIMD
backends. `conv1d` and `rmsnorm` are deliberately simple loops; together they
are under 3% of a layer, too little to dominate the profile.

The scan stores its state in a $[t][n][c]$ layout so that the inner loop over channels is unit-stride, which turns into contiguous vector loads over full cache lines. A $[t][c][n]$ layout would stride the reduction and waste bandwidth on partial lines. See [ADR 0001](../decisions/0001-cpu-kernels).

## Threading

To saturate memory bandwidth, large matrix multiplications dispatch concurrently
across an SPMC lock-free thread pool. Workers spin briefly (`_mm_pause` / `yield`)
to absorb gaps inside a forward pass, then park with C++ `atomic::wait`. Enqueue
increments a work epoch and wakes a worker, avoiding both a lost-wakeup race and
permanently burning CPU between inference requests. Threads are pinned to CPU
cores where the platform allows it. The task ring and worker storage are
arena-carved, power-of-two sized, and isolated on destructive-interference
boundaries.

Pool capacity and per-kernel parallelism are separate. The runtime defaults to at most four
workers, while an explicit override can provision up to eight. Each matrix uses a C++23
power-of-two task selector based on rows, input width, output width, and F32 versus BF16 work.
Small projections stay on the caller thread; large projections use only the useful 2/4/8 tier.
This avoids dispatch-dominated kernels and SMT/virtualization oversubscription.

## BF16 Weight Widening

`in_proj`, `out_proj`, and Mamba projection weights may be stored as `BF16` in
the arena. The loader exposes a typed `WeightView`, and model code dispatches
through `weight_ops.h`, so F32 and BF16 weights share one high-level call.
BF16 is widened to `F32` inline during the innermost SIMD loop of `matmul`.

This doubles effective DRAM bandwidth at the negligible cost of zero-extension and shift instructions (`_mm256_castsi256_ps` on AVX2, or `vreinterpretq_f32_u32` on NEON).

Because the model only ever calls this header, a new backend is the same ten functions and a link-time switch.
