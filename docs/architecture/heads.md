# Heads

A head takes a conditioning vector and a noise sample and returns an action chunk. It is backbone-agnostic: any backbone that produces a vector of the right size can drive any head.

## Flow matching

The head is a velocity field $v_\theta(x, t \mid c)$, and sampling integrates an ODE from noise to action:

$$
\frac{dx}{dt} = v_\theta(x, t \mid c), \qquad x(0) = x_0 \sim \mathcal{N}(0, I), \qquad \text{action} = x(1)
$$

Training regresses $v_\theta$ onto the velocity of a straight path between noise and data, $x_t = (1-t)\,x_0 + t\,x_1$, whose target velocity is the constant $x_1 - x_0$. Sampling then integrates with a fixed number of function evaluations:

$$
x_{t+\Delta} = x_t + \Delta\, v_\theta(x_t, t \mid c) \qquad \text{(Euler)}
$$

```{mermaid}
%%{init: {'theme':'base','flowchart':{'htmlLabels':false,'nodeSpacing':28,'rankSpacing':34,'useMaxWidth':false},'themeVariables':{'primaryColor':'#f6ead0','primaryBorderColor':'#7b2733','lineColor':'#7b2733','primaryTextColor':'#2b2521','secondaryColor':'#eaddbf','tertiaryColor':'#faf3e2','fontFamily':'system-ui, -apple-system, Segoe UI, Roboto, sans-serif','fontSize':'13px'}}}%%
graph LR
  Z["x = noise"] --> V["v(x, t, cond)"]
  V --> STEP["x += dt mul v"]
  STEP -->|"next step"| V
  STEP -->|"t = 1"| OUT[action]
```

Heun uses two evaluations per step and RK4 uses four. The conditioning $c$ is projected once and reused across every step.

## Why this way

The head's cost comes down to one number, the count of velocity-net evaluations, or NFE, since the total time is NFE times a single net eval. Euler spends one eval per step, Heun two, RK4 four. A deterministic flow ODE reaches the action in roughly ten evals, where a diffusion head denoises over dozens of stochastic steps. On a fixed loop budget that gap decides whether the step lands inside the period, which is why flow matching is the first head.

Solver order is a second lever on the same budget. A higher-order step like RK4 cuts the discretization error of each step, so the same accuracy needs fewer of them. You trade evals per step against step count and pick whatever fits.

Because $c$ is projected once and held fixed for the whole integration, the backbone and prefix cost is paid once per action rather than once per step, and only the small velocity net runs inside the loop. The loop carries no RNG, so the same observation always yields the same action, which is what a controller needs.

Source: `src/heads/flow/`. See [ADR 0004](../decisions/0004-decouple-head).

## Diffusion, ACT, others

Planned, behind the same contract. Diffusion denoises instead of integrating a velocity field. See the [head guide](../guides/add-a-head) and the [roadmap](../roadmap).
