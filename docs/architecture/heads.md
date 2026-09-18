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

```{image} ../_static/figures/flow-matching.svg
:alt: Flow matching ODE from noise x0 to action x1
:class: fe-fig
```

Heun uses two evaluations per step and RK4 uses four. Conditioning $c$ is projected once and reused.

The flow head can be loaded without a backbone and driven directly by an external condition vector.
That makes the solver usable behind vision-language-action models, observation encoders, and model
servers that should not be linked into the core engine.

Sampling also has a resumable form. `sampler_begin` stores the projected condition, current action,
and Runge-Kutta stages in caller-owned fixed workspace. `sampler_advance` runs a bounded number of
complete ODE steps. Splitting a solve does not change floating-point operation order, so the final
action is bit-identical to a monolithic solve on the same backend. With `FLOWEDGE_BACKEND=cuda`,
weights and ODE scratch stay on device after load; the public spans stay host pointers.

## Why this way

A robot period is a budget of velocity-net evaluations (NFE). Flow matching is a straight-path ODE: about ten evals, no RNG, condition \(c\) projected once. A diffusion head often spends dozens of stochastic steps on the same action. That is why flow matching is the default, and why DP is the other first-class head only when that is the trained checkpoint.

PyTorch can run both. It does not give you a slab sized at load or a miss contract. Split at solver steps (`flow_begin` / `flow_advance`), never inside a matmul. [Cooperative execution](cooperative-execution).

Source: `src/core/heads/flow/`. [ADR 0004](../decisions/0004-decouple-head).

## Diffusion Policy

The head is an epsilon network \(\epsilon_\theta(x_t, t \mid c)\) on a noisy
action horizon \(x_t\). Training is the usual noise-prediction objective.
Sampling inverts that process. **DDIM** is a deterministic reverse walk (no
RNG; NFE equals the step count). **DDPM** keeps the stochastic term and is
seeded for reference. After the last step, checkpoint MIN_MAX statistics map
the horizon back to dataset action units. Each DDIM step is one U-Net forward
plus a closed-form scheduler update; \(c\) is reused. With `FLOWEDGE_BACKEND=cuda`,
weights and U-Net/DDIM scratch stay on device after load; the public spans stay
host pointers.

The fixed-shape `ConditionalUnet1D` is the LeRobot Conv1D U-Net: GroupNorm,
Mish, sinusoidal timestep embedding, FiLM scale/bias. Observation encoding
stays outside Core so the same head can sit behind LeRobot, a VLA, or
TensorRT. See the [Diffusion Policy guide](../guides/diffusion-policy).

## ACT and others

Planned, behind the same contract. See the [head guide](../guides/add-a-head) and
the [roadmap](../roadmap).
