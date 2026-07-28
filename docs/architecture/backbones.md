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

It is selective, meaning $\Delta_t$, $B_t$, and $C_t$ are functions of the input $x_t$, and $A = -\exp(A_{\log})$ is diagonal per state and channel. Each layer runs:

```{mermaid}
%%{init: {'theme':'base','flowchart':{'htmlLabels':false,'nodeSpacing':28,'rankSpacing':34,'useMaxWidth':false},'themeVariables':{'primaryColor':'#f6ead0','primaryBorderColor':'#7b2733','lineColor':'#7b2733','primaryTextColor':'#2b2521','secondaryColor':'#eaddbf','tertiaryColor':'#faf3e2','fontFamily':'system-ui, -apple-system, Segoe UI, Roboto, sans-serif','fontSize':'13px'}}}%%
graph LR
  X[hidden] --> N[RMSNorm]
  N --> IP[in_proj]
  IP --> CV[conv1d]
  CV --> SL[SiLU]
  SL --> XP[x_proj]
  XP --> DT["dt_proj + softplus"]
  XP --> BC["B, C"]
  DT --> DS[discretize]
  BC --> DS
  DS --> SS[selective_scan]
  SS --> G["gate: y mul silu(z)"]
  G --> OP[out_proj]
  OP --> R["+ residual"]
```

Config comes from the checkpoint shapes, and the state is $d_{inner} \times d_{state}$. Streaming decode keeps a conv window and $h$ per layer and advances one step per token.

## Why this way

Picking a backbone is really picking how memory scales. Attention costs $O(L^2 d)$ compute and an $O(L d)$ KV cache that grows with the sequence, whereas the SSM recurrence costs $O(L\, d_{inner} d_{state})$ compute and carries an $O(d_{inner} d_{state})$ state that is constant in $L$. A control loop has an open-ended horizon, and an embedded target cannot let memory grow with time, so constant state is the property that decides it. The arena could not bound a cache that grows either.

The diagonal $A$ keeps the state update elementwise. $\bar{A} h_{t-1}$ is a Hadamard product rather than a matmul, so a step is $O(d_{inner} d_{state})$ and vectorizes cleanly. Letting $\Delta, B, C$ depend on $x_t$ is what a fixed convolution cannot do, and it is what lets the model track a changing observation.

Discretization is materialized, not fused. Writing $\bar{A}$ and $\bar{B}u$ out as $[t][n][c]$ adds memory traffic, which is the scan's dominant cost, but it keeps the kernel simple and checkable against PyTorch. Fusing it into the scan would cut that traffic at the price of changed numerics, so it waits behind the ULP gate. For now the trade is bandwidth for verifiability.

Since the recurrence carries the same state over a batch or a single token, streaming decode is one step. It tracks the batch forward to about 2e-7, the two paths differing only in floating-point ordering.

Source: `src/models/mamba/`. See [ADR 0002](../decisions/0002-mamba-block).

## Transformer

Planned, behind the same contract: attention plus MLP, with a KV-cache sized to the max prefix so the state stays fixed.
