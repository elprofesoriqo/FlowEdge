# Backbones

A backbone reads the token prefix and returns a conditioning vector, the last hidden state. It also runs O(1) streaming decode.

## Mamba

Mamba is a selective state-space model. At its core is a linear recurrence over a hidden state $h$, written in continuous form as:

$$
h'(t) = A\,h(t) + B\,x(t), \qquad y(t) = C\,h(t)
$$

With a step size $\Delta$ this discretizes into the recurrence the engine runs directly:

$$
\bar{A} = \exp(\Delta A), \quad \bar{B} = \Delta B, \quad
h_t = \bar{A}\,h_{t-1} + \bar{B}\,x_t, \quad y_t = C_t\,h_t + D\,x_t
$$

It is selective, meaning $\Delta_t$, $B_t$, and $C_t$ are functions of the input $x_t$, and $A = -\exp(A_{\log})$ is diagonal per state and channel. Each layer:

```{image} ../_static/figures/mamba.svg
:alt: Mamba layer RMSNorm, in_proj, conv1d, SiLU, scan, gate, residual
:class: fe-fig
```

Config comes from the checkpoint shapes, and the state is $d_{inner} \times d_{state}$. Streaming
decode keeps a conv window and $h$ per layer and advances one step per token. The batch path clears
its temporary recurrence; the streaming path deliberately preserves the caller-owned state. That
distinction is part of the kernel contract and is checked by batch-versus-streaming tests.

## Why this way

Attention is $O(L^2 d)$ with a growing KV cache. The SSM is $O(L\, d_{inner} d_{state})$ with state that does not grow with $L$. Diagonal $A$ makes $\bar{A} h_{t-1}$ a Hadamard product. `discretize_and_scan` fuses the rest; $-\exp(A_{\log})$ is transposed once at load. Streaming decode is one step of the same recurrence.

Source: `src/core/models/mamba/`. [ADR 0002](../decisions/0002-mamba-block).

## Transformer

CPU decoder baseline: fused QKV, learned positions, KV cache sized to
`max_sequence`, `cached_causal_attention` in `kernels.h`. Kernel and fixture
gtests plus the tiny-gpt2 conversion artifact. Not a policy. See
[Fixed-shape Transformer baseline](../guides/transformer-backbone).
