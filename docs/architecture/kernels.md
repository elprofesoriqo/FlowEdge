# Kernels

One header declares every kernel, and each backend implements it.

```{mermaid}
%%{init: {'theme':'base','flowchart':{'htmlLabels':false,'nodeSpacing':28,'rankSpacing':34,'useMaxWidth':false},'themeVariables':{'primaryColor':'#f6ead0','primaryBorderColor':'#7b2733','lineColor':'#7b2733','primaryTextColor':'#2b2521','secondaryColor':'#eaddbf','tertiaryColor':'#faf3e2','fontFamily':'system-ui, -apple-system, Segoe UI, Roboto, sans-serif','fontSize':'13px'}}}%%
graph TD
  I["kernels.h"]
  I --> CPU["cpu: common + avx2 / neon / scalar"]
  I --> CU["cuda (planned)"]
  I --> TT["tenstorrent (planned)"]
```

The backend is picked at build time:

```bash
cmake -B build -DFLOWEDGE_BACKEND=cpu
```

The CPU backend splits by ISA. `kernels_common.cc` holds `conv1d_causal`, `conv1d_step` and `rmsnorm`, which stay scalar everywhere. `kernels_avx2.cc`, `kernels_neon.cc` and `kernels_scalar.cc` each implement the other seven, and `CMAKE_SYSTEM_PROCESSOR` picks one. Every kernel takes `std::span`, owns no memory, and never allocates, over 64-byte-aligned caller buffers.

The set is `matmul`, `conv1d_causal`, `conv1d_step`, `rmsnorm`, `silu`, `softplus`, `gate_silu`, `selective_scan`, `scan_step`, and `discretize`.

## The exp and log kernels

`exp` and `log` use the Cephes method, because the compiler folds a scalar `exp` back into a per-element libm call. `exp` reduces its argument to a small interval and then evaluates a polynomial:

$$
n = \left\lfloor x \log_2 e \right\rceil, \qquad r = x - n \ln 2, \qquad \exp(x) = 2^{n}\, P(r)
$$

$P$ is a degree-5 minimax polynomial on $r \in [-\tfrac{\ln 2}{2}, \tfrac{\ln 2}{2}]$, and $2^{n}$ is applied by writing $n$ into the float exponent. The result is within about 1 ULP. The vector forms are `exp8` on AVX2 and `exp4` on NEON.

## Why this way

Almost everything here is bandwidth-bound. That sets the optimization order.

`matmul` dominates the runtime, about 84 percent of a layer. A square GEMM would be compute-bound, with $O(N^3)$ FLOPs over $O(N^2)$ bytes, but that is not the regime a policy runs in. The prefix is short, so each weight is read only a handful of times and the arithmetic intensity falls to roughly $2\,\text{rows}$ FLOPs per byte, well under the machine balance. Measurement agrees: `in_proj` and `out_proj` have very different shapes and both land near 15 GB/s, which is a DRAM ceiling rather than an FMA one, while the small cache-resident `x_proj` runs half again faster. So the lever on `matmul` is weight traffic, not FMA blocking. Storing weights in BF16 halves the bytes, INT8 quarters them, and threading buys more memory parallelism. See [ADR 0007](../decisions/0007-roofline).

`selective_scan` is bandwidth-bound too, and small at 2.5% of a layer. It streams $\bar{A}$ and $\bar{B}u$ once with almost no reuse, so intensity is near one FLOP per byte and the ALU waits on memory. Tuning it further gains nothing.

`matmul`, the activations, `discretize` and the scan step carry explicit intrinsics. `conv1d` and `rmsnorm` stay scalar in `kernels_common.cc`: together they are under 3% of a layer, too little to pay for three ISA copies.

The scan stores its state in a $[t][n][c]$ layout so that the inner loop over channels is unit-stride, which turns into contiguous vector loads over full cache lines. A $[t][c][n]$ layout would stride the reduction and waste bandwidth on partial lines. See [ADR 0001](../decisions/0001-cpu-kernels).

Because the model only ever calls this header, a new backend is the same ten functions and a link-time switch.
