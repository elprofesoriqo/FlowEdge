# FlowEdge

<img class="fe-hero-img" src="_static/hero.png" alt="FlowEdge" />

<p class="fe-lede">FlowEdge is a C++23 inference runtime for robotics policies with fixed memory, deterministic execution, and explicit deadline handling. It runs two heads: <strong>flow matching</strong> (a short ODE from noise to action) and <strong>Diffusion Policy</strong> (DDIM on a Conv1D U-Net). LeRobot still owns training, cameras, and the robot driver.</p>

```{raw} html
<ul class="fe-chips">
  <li>C++23 Core</li>
  <li>CPU · CUDA</li>
  <li>Tenstorrent preview</li>
  <li>LeRobot adapter</li>
</ul>

<div class="fe-metrics">
  <div class="fe-metric">
    <span class="fe-metric-value">851 vs 1409</span>
    <span class="fe-metric-label">CPU DDIM p50 · ms · threads=1</span>
  </div>
  <div class="fe-metric">
    <span class="fe-metric-value">131 vs 345</span>
    <span class="fe-metric-label">CUDA DDIM p50 · ms · GTX 1650</span>
  </div>
  <div class="fe-metric">
    <span class="fe-metric-value">malloc = 0</span>
    <span class="fe-metric-label">Supported hot paths after load</span>
  </div>
</div>

<nav class="fe-jump" aria-label="Start here">
  <a class="fe-jump-card" href="getting-started.html">
    <span class="fe-jump-kicker">01</span>
    <strong>Get started</strong>
    <span>Build Core, convert a mix, run the first sample.</span>
  </a>
  <a class="fe-jump-card" href="python-quickstart.html">
    <span class="fe-jump-kicker">02</span>
    <strong>Python</strong>
    <span>Wheel, Engine, predict, and DDIM from CPython.</span>
  </a>
  <a class="fe-jump-card" href="guides/lerobot.html">
    <span class="fe-jump-kicker">03</span>
    <strong>LeRobot</strong>
    <span>Plugin, period, and <code>--on-miss</code> around the native head.</span>
  </a>
  <a class="fe-jump-card" href="performance.html">
    <span class="fe-jump-kicker">04</span>
    <strong>Performance</strong>
    <span>Fair compare vs PyTorch. Not a 10 ms loop.</span>
  </a>
  <a class="fe-jump-card" href="architecture/overview.html">
    <span class="fe-jump-kicker">05</span>
    <strong>Architecture</strong>
    <span>Arena, kernels, CUDA residency, two product heads.</span>
  </a>
  <a class="fe-jump-card" href="api/c-abi.html">
    <span class="fe-jump-kicker">06</span>
    <strong>C API</strong>
    <span><code>fe_engine_*</code> load, sample, and miss policy.</span>
  </a>
</nav>
```

## News

**September 2026**

```{raw} html
<ul class="fe-news">
  <li class="fe-news-item">
    <span class="fe-news-date">19</span>
    <p>Split-K CUDA Conv1D on horizon 8 (<a href="https://github.com/elprofesoriqo/FlowEdge/pull/183">#183</a>). GTX 1650 matched DDIM replay stays <strong>131 vs 345 ms</strong> vs PyTorch CUDA. Not a 10 ms loop.</p>
  </li>
  <li class="fe-news-item">
    <span class="fe-news-date">18</span>
    <p>Device-resident CUDA after load: flow (<a href="https://github.com/elprofesoriqo/FlowEdge/pull/167">#167</a>), Diffusion Policy (<a href="https://github.com/elprofesoriqo/FlowEdge/pull/168">#168</a>, <a href="https://github.com/elprofesoriqo/FlowEdge/pull/169">#169</a>), Mamba (<a href="https://github.com/elprofesoriqo/FlowEdge/pull/173">#173</a>). Same-process CUDA vs PyTorch CUDA (<a href="https://github.com/elprofesoriqo/FlowEdge/pull/170">#170</a>). <code>--device cuda</code> period rollouts (<a href="https://github.com/elprofesoriqo/FlowEdge/pull/171">#171</a>).</p>
  </li>
  <li class="fe-news-item">
    <span class="fe-news-date">17</span>
    <p><a href="https://github.com/elprofesoriqo/FlowEdge/releases/tag/v0.1.1">v0.1.1</a> CPU wheels (manylinux, Windows, macOS ARM). Not on PyPI. <code>--on-miss hold|drop|raise</code> (<a href="https://github.com/elprofesoriqo/FlowEdge/pull/152">#152</a>). CPU fair-compare <strong>851 vs 1409 ms</strong>, <code>threads=1</code>.</p>
  </li>
</ul>
```

<p class="fe-inline-links"><a href="https://github.com/elprofesoriqo/FlowEdge/releases">GitHub Releases</a> · <a href="performance.html">Performance</a> · <a href="https://github.com/elprofesoriqo/FlowEdge">Source</a></p>

## The problem

A robot loop is a deadline. At 100 Hz the motors want a new command every 10 ms. If this sample is late, they still need a defined action — hold, skip, or fault — and that choice has to be replayable (`--on-miss`). Matched Diffusion Policy replay is faster than PyTorch: CPU **851 vs 1409 ms**, CUDA **131 vs 345 ms**. That is the fair compare. The 10 ms figure is the robot’s tick, not this checkpoint’s p50.

The usual deploy path is “export the training graph and hope”: PyTorch or a general compiler, heap traffic after warmup, a dispatcher, sometimes a KV cache that grows with the horizon. That stack is the right tool for training and for broad model zoos. It is the wrong contract for a period loop with a replayable miss policy.

```{image} _static/matched_replay.gif
:alt: Matched replay loading, CUDA then CPU. FlowEdge CUDA 131 ms vs PyTorch 345 ms; FlowEdge CPU 851 ms vs PyTorch 1409 ms.
:class: fe-fig
```

## How FlowEdge solves it

1. **Convert once.** Map a pinned Hugging Face / LeRobot checkpoint into layouts the engine already knows (`backbone.*`, `flow.*`, `dp.*`).
2. **Load into one arena.** Weights, decode state, ODE scratch, and the thread pool are carved from a slab sized at load. Supported hot paths do not call `malloc`.
3. **Sample the head.** Flow matching integrates \(dx/dt = v(x,t \mid c)\) from noise to action (Euler / Heun / RK4). Diffusion Policy denoises an action horizon with DDIM; the RGB encoder stays outside Core. Same miss contract for both.
4. **Honor the period.** `--period-ms` plus `--on-miss hold|drop|raise`. Limits and e-stop are not in this library.

```{image} _static/figures/purpose.svg
:alt: Hardware runs the loop, FlowEdge runs the policy, LeRobot trains and drives the robot
:class: fe-fig
```

```{image} _static/figures/workflow.svg
:alt: convert, load arena, sample flow or DDIM, period, act
:class: fe-fig
```

[Architecture](architecture/overview) · [Relay](guides/relay-quickstart) · [Flow matching vs DDIM](architecture/heads)

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

```{toctree}
:hidden:
:maxdepth: 1
:caption: Start
Overview <self>
getting-started
python-quickstart
guides/lerobot
performance
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Guides
guides/converter
guides/diffusion-policy
guides/transformer-backbone
guides/model-import
guides/checkpoint-preflight
guides/verification
guides/policy-evaluation
guides/edge-benchmarks
guides/relay-quickstart
guides/action-delivery
guides/deadline-flow
guides/deadline-profile
guides/observability
guides/cooperative-jobs
guides/generic-job-daemon
guides/sanitizers
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Architecture
architecture/overview
architecture/memory
architecture/loader
architecture/kernels
architecture/cuda
architecture/backbones
architecture/heads
architecture/deployment-profile
architecture/cooperative-execution
architecture/model-porting
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: API
api/c-abi
api/python
capabilities
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Contribute
contributing
guides/add-a-head
guides/add-a-backbone
benchmarks
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Project
roadmap
product-direction
tenstorrent-program
decisions/index
FlowEdge Relay <ecosystem/relay-proposal>
LeRobot edge-inference RFC draft <ecosystem/lerobot-rfc>
```
