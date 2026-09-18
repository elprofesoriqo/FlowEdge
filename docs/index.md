# FlowEdge

<img class="fe-hero-img" src="_static/hero.png" alt="FlowEdge" />

<p class="fe-lede">FlowEdge is a C++23 inference engine for two real-time robotics policy heads: <strong>flow matching</strong> (a short ODE from noise to action) and <strong>Diffusion Policy</strong> (DDIM on a Conv1D U-Net). It loads a converted checkpoint into a fixed memory arena and produces an action chunk under a control period. LeRobot still owns training, cameras, and the robot driver.</p>

## The problem

A manipulator or mobile base does not care about median throughput. It cares whether *this* cycle finished before the next tick. If the policy is late, the controller still has to send an action, and you have to be able to explain which action that was.

The usual deploy path is “export the training graph and hope”: PyTorch or a general compiler, heap traffic after warmup, a dispatcher, sometimes a KV cache that grows with the horizon. That stack is the right tool for training and for broad model zoos. It is the wrong contract for a 10 ms loop with a replayable miss policy.

## How FlowEdge solves it

1. **Convert once.** Map a pinned Hugging Face / LeRobot checkpoint into layouts the engine already knows (`backbone.*`, `flow.*`, `dp.*`).
2. **Load into one arena.** Weights, decode state, ODE scratch, and the thread pool are carved from a slab sized at load. Supported hot paths do not call `malloc`.
3. **Sample the head.** Flow matching integrates \(dx/dt = v(x,t \mid c)\) from noise to action (Euler / Heun / RK4). Diffusion Policy denoises an action horizon with DDIM; the RGB encoder stays outside Core. Same miss contract for both.
4. **Honor the period.** `--period-ms` plus `--on-miss hold|drop|raise`. Limits and e-stop are not in this library.

```{image} _static/figures/purpose.svg
:alt: Hardware runs the loop, FlowEdge runs the policy, LeRobot trains and drives the robot
:class: fe-fig
```

```{image} _static/figures/big-flow.svg
:alt: Whole FlowEdge flow from checkpoint to motors
:class: fe-fig
```

```{image} _static/figures/relay.svg
:alt: Optional Relay flow from sensor or VLA through shared-memory rings to Core
:class: fe-fig
```

## Why not the alternatives

```{image} _static/figures/why-this.svg
:alt: Instead of PyTorch-in-the-loop, TensorRT/ONNX, a growing KV cache, long DDIM, or model servers — FlowEdge runs this flow or DP head in a sized arena with a miss contract
:class: fe-fig
```

| Tool | Use it for | Not as |
|---|---|---|
| LeRobot / PyTorch | Train, encode RGB, drive the robot, compare parity | The malloc-free period loop |
| TensorRT, ONNX, ExecuTorch | General DAGs on a given backend | This flow or DP head’s arena, miss contract, and native kernels |
| LLM / VLA servers | Throughput, batching, GPUs | Newest valid action before a physical deadline |
| A Transformer KV cache | Language and long context | An open-ended robot horizon on a bounded RSS |

Flow matching is the default head because a short deterministic ODE is cheaper, in NFE, than a long DDIM walk. Diffusion Policy is first-class when that is the trained checkpoint. Mamba is the default backbone for the flow path because its state does not grow with time.

```{image} _static/figures/policies.svg
:alt: Flow matching ODE versus Diffusion Policy DDIM
:class: fe-fig
```

```{image} _static/flowedge.gif
:alt: Flow matching Euler steps from noise to action, malloc = 0
:class: fe-fig
```

Matched PushT **Diffusion Policy** replay, CPU, `threads=1`: policy p50 **851 ms** vs LeRobot/PyTorch **1409 ms**. Not a 10 ms loop. **Flow matching** matches the same PyTorch reference on ULP (~1e-6 rel); that is not a policy p50. [Performance](performance).

```{image} _static/perf.gif
:alt: Matched PushT replay bars, FlowEdge 851 ms vs PyTorch 1409 ms
:class: fe-fig
```

```{image} _static/figures/workflow.svg
:alt: convert, load arena, sample flow or DDIM, period, act
:class: fe-fig
```

Not a trainer. Not a graph runtime. Not a safety controller.

## What you can use today

| Goal | Entry point |
|---|---|
| Run flow matching | [Getting Started](getting-started) |
| Deploy LeRobot Diffusion Policy | [LeRobot adapter](guides/lerobot) |
| Numbers vs PyTorch | [Performance](performance) |
| Convert a checkpoint | [Converter](guides/converter) |
| Keep your encoder, run only the head | [Capabilities](capabilities) |
| Optional local IPC + deadlines | [Relay quickstart](guides/relay-quickstart) |
| Cached SmolVLA expert (VLM in LeRobot) | [Transformer / SmolVLA](guides/transformer-backbone) |

## Implementation status

```{image} _static/figures/status.svg
:alt: Backends, backbones, heads, models, precisions
:class: fe-fig
```

```{toctree}
:hidden:
:caption: Start
Overview <self>
getting-started
capabilities
guides/lerobot
guides/policy-evaluation
performance
benchmarks
```

```{toctree}
:hidden:
:caption: Architecture
architecture/overview
architecture/memory
architecture/loader
architecture/deployment-profile
architecture/kernels
architecture/cuda
architecture/backbones
architecture/heads
architecture/cooperative-execution
architecture/model-porting
decisions/index
```

```{toctree}
:hidden:
:caption: API
api/c-abi
api/python
```

```{toctree}
:hidden:
:caption: Guides
guides/add-a-head
guides/add-a-backbone
guides/converter
guides/checkpoint-preflight
guides/diffusion-policy
guides/transformer-backbone
guides/model-import
guides/edge-benchmarks
guides/deadline-profile
guides/sanitizers
guides/verification
guides/observability
guides/cooperative-jobs
guides/relay-quickstart
guides/action-delivery
guides/generic-job-daemon
```

```{toctree}
:hidden:
:caption: Reference
roadmap
product-direction
guides/deadline-flow
tenstorrent-program
FlowEdge Relay <ecosystem/relay-proposal>
LeRobot edge-inference RFC draft <ecosystem/lerobot-rfc>
```
