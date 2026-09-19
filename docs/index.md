# FlowEdge

<img class="fe-hero-img" src="_static/hero.png" alt="FlowEdge" />

<p class="fe-lede">FlowEdge is a C++23 inference runtime for robotics policies with fixed memory, deterministic execution, and explicit deadline handling. It runs two heads: <strong>flow matching</strong> (a short ODE from noise to action) and <strong>Diffusion Policy</strong> (DDIM on a Conv1D U-Net). LeRobot still owns training, cameras, and the robot driver.</p>

```{image} _static/matched_replay.gif
:alt: Matched replay loading, CUDA then CPU. FlowEdge CUDA 131 ms vs PyTorch 345 ms; FlowEdge CPU 851 ms vs PyTorch 1409 ms.
:class: fe-fig
```

## 📰 News

**🗓️ September 2026**

- ⚡ **19** — Split-K CUDA Conv1D on horizon 8 ([#183](https://github.com/elprofesoriqo/FlowEdge/pull/183)). GTX 1650 matched DDIM replay stays **131 vs 345 ms** vs PyTorch CUDA. Not a 10 ms loop.
- 🚀 **18** — Device-resident CUDA after load: flow ([#167](https://github.com/elprofesoriqo/FlowEdge/pull/167)), Diffusion Policy ([#168](https://github.com/elprofesoriqo/FlowEdge/pull/168), [#169](https://github.com/elprofesoriqo/FlowEdge/pull/169)), Mamba ([#173](https://github.com/elprofesoriqo/FlowEdge/pull/173)). Same-process CUDA vs PyTorch CUDA ([#170](https://github.com/elprofesoriqo/FlowEdge/pull/170)). `--device cuda` period rollouts ([#171](https://github.com/elprofesoriqo/FlowEdge/pull/171)).
- 📦 **17** — [v0.1.1](https://github.com/elprofesoriqo/FlowEdge/releases/tag/v0.1.1) CPU wheels (manylinux, Windows, macOS ARM). Not on PyPI. ⏱️ `--on-miss hold|drop|raise` ([#152](https://github.com/elprofesoriqo/FlowEdge/pull/152)). CPU fair-compare **851 vs 1409 ms**, `threads=1`.

🏷️ [GitHub Releases](https://github.com/elprofesoriqo/FlowEdge/releases) · 📈 [Performance](performance)

## The problem

A robot loop is a deadline. At 100 Hz the motors want a new command every 10 ms. If this sample is late, they still need a defined action — hold, skip, or fault — and that choice has to be replayable (`--on-miss`). Matched Diffusion Policy replay is faster than PyTorch: CPU **851 vs 1409 ms**, CUDA **131 vs 345 ms**. That is the fair compare. The 10 ms figure is the robot’s tick, not this checkpoint’s p50.

The usual deploy path is “export the training graph and hope”: PyTorch or a general compiler, heap traffic after warmup, a dispatcher, sometimes a KV cache that grows with the horizon. That stack is the right tool for training and for broad model zoos. It is the wrong contract for a period loop with a replayable miss policy.

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

```{image} _static/flowedge.gif
:alt: Flow matching Euler steps from noise to action, malloc = 0
:class: fe-fig
```

```{image} _static/figures/policies.svg
:alt: Flow matching ODE versus Diffusion Policy DDIM
:class: fe-fig
```

```{image} _static/figures/workflow.svg
:alt: convert, load arena, sample flow or DDIM, period, act
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

Matched PushT **Diffusion Policy** replay, CPU, `threads=1`: policy p50 **851 ms** vs LeRobot/PyTorch **1409 ms**. Not a 10 ms loop. On GTX 1650 the same matched replay is **131 ms** vs PyTorch CUDA **345 ms**. **Flow matching** matches the same PyTorch reference on ULP (~1e-6 rel); that is not a policy p50. [Performance](performance).

Not a trainer. Not a graph runtime. Not a safety controller.

## What you can use today

| Goal | Entry point |
|---|---|
| Run flow matching | [Getting Started](getting-started) |
| Install the Python wheel | [Python](python-quickstart) |
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
:caption: Use
Overview <self>
getting-started
python-quickstart
guides/lerobot
performance
guides/converter
```

```{toctree}
:hidden:
:caption: Contribute
contributing
guides/add-a-head
guides/add-a-backbone
guides/verification
guides/edge-benchmarks
guides/policy-evaluation
benchmarks
```

```{toctree}
:hidden:
:caption: Internals
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
capabilities
api/c-abi
api/python
guides/checkpoint-preflight
guides/diffusion-policy
guides/transformer-backbone
guides/model-import
guides/deadline-profile
guides/sanitizers
guides/observability
guides/cooperative-jobs
guides/relay-quickstart
guides/action-delivery
guides/generic-job-daemon
roadmap
product-direction
guides/deadline-flow
tenstorrent-program
FlowEdge Relay <ecosystem/relay-proposal>
LeRobot edge-inference RFC draft <ecosystem/lerobot-rfc>
```
