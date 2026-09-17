<div align="center">

<img src="assets/surfingtux.png" width="600" alt="FlowEdge">

# FlowEdge

**Low-latency inference for real-time robotics policies**

A C++23 engine that loads a converted **flow-matching or Diffusion Policy** into a fixed memory arena and returns an action chunk inside a control period. LeRobot still owns cameras, training, and the robot.

<br/>

**CPU available** &nbsp;&nbsp;·&nbsp;&nbsp; <img alt="Tenstorrent / TT-Metal" height="22" src="assets/tenstorrent.jpg">&nbsp;**Tenstorrent planned** &nbsp;&nbsp;·&nbsp;&nbsp; <img alt="NVIDIA CUDA" height="22" src="assets/nvidia.svg">&nbsp;**CUDA planned**

[Documentation](https://elprofesoriqo.github.io/FlowEdge/) &nbsp;&nbsp;·&nbsp;&nbsp; [Getting Started](https://elprofesoriqo.github.io/FlowEdge/getting-started.html) &nbsp;&nbsp;·&nbsp;&nbsp; [Performance](https://elprofesoriqo.github.io/FlowEdge/performance.html) &nbsp;&nbsp;·&nbsp;&nbsp; [Contributing](CONTRIBUTING.md)

</div>

***

## The problem

A robot loop is a deadline. At 100 Hz you have 10 ms to observe, infer, and send an action. If inference overruns, you still owe the motors *something* — last action, skip, or fault — and you must be able to replay that choice.

PyTorch, LeRobot, TensorRT, and generic model servers are good at training or at running arbitrary graphs. They are a weak fit as the thing inside that loop: heap allocations after warmup, a dispatcher, a growing KV cache, and a p50 that is not a miss count. Training code also tends to own cameras, normalization, and e-stop, which do not belong in the inference kernel.

## What FlowEdge does

Convert a pinned checkpoint once. Load it into one bump-allocated arena (weights, scratch, solver state). On each tick the engine samples **one of two policy heads**:

**Flow matching** — Mamba (or your encoder) produces a condition \(c\). The head is a velocity field \(v(x,t \mid c)\). Integrate a short ODE from noise \(x_0\) to action \(x_1\) with Euler / Heun / RK4. Cost is NFE × one velocity net. No RNG.

**Diffusion Policy** — LeRobot still encodes RGB. Core runs the Conv1D U-Net and DDIM (seeded DDPM for reference) on the action horizon. Cost is DDIM steps × one U-Net. Usually more NFE than a short flow ODE, which is why flow matching is the default and DP is the other first-class head when that is the trained checkpoint.

The engine sizes RSS at load and does not malloc on the supported hot path. Same observation, same action. If the period is missed, `--on-miss hold|drop|raise` is explicit. Encoders, joint limits, and e-stop stay in LeRobot (or your adapter). Optional Relay is local IPC around Core, not the default DP path.

<div align="center">
  <img src="assets/flowedge.gif" width="720" alt="Flow matching: Euler steps from noise x0 to action x1, malloc = 0">
</div>

## Why this, not the usual stack

<div align="center">
  <img src="docs/_static/figures/why-this.svg" width="920" alt="Instead of PyTorch-in-the-loop, TensorRT/ONNX, a growing KV cache, long DDIM, or model servers: FlowEdge runs this flow or DP head in a sized arena with a miss contract. Not a trainer, graph compiler, or safety controller. DP p50 851 vs 1409 ms on CPU, not a 10 ms loop.">
</div>

<div align="center">
  <img src="assets/perf.gif" width="720" alt="Matched PushT replay: FlowEdge policy p50 851 ms vs PyTorch 1409 ms, threads=1">
</div>

Matched **Diffusion Policy** PushT replay, CPU, `threads=1`: policy p50 **851 ms** vs LeRobot/PyTorch **1409 ms**. It is not a 10 ms loop. **Flow matching** is gated by ULP vs the same PyTorch reference (~1e-6 rel), not a published policy p50. Evidence: [performance](docs/performance.md).

## Everyday paths

```bash
# Flow matching
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/flow_sample models/mamba_flow.safetensors euler 10

# LeRobot Diffusion Policy (plugin + fake robot or SO-100 shim)
python -m pip install . && python -m pip install -e integrations/lerobot
python -m flowedge_dev pipeline convert models/diffusion_pusht \
  models/diffusion_pusht.flowedge.safetensors --arch diffusion
python -m flowedge_dev pipeline rollout models/diffusion_pusht.flowedge.safetensors \
  --steps 20 --threads 4 --period-ms 10 --on-miss hold

# Relay (optional IPC — not on the DP replay path)
cmake -S . -B build-relay -DCMAKE_BUILD_TYPE=Release -DFLOWEDGE_RELAY=ON
cmake --build build-relay -j
FLOWEDGE_BUILD_DIR=build-relay ./scripts/relay_demo.sh models/mamba_flow.safetensors
```

<div align="center">
  <img src="docs/_static/figures/purpose.svg" width="920" alt="Hardware runs the loop, FlowEdge runs the policy, LeRobot trains and drives the robot">
</div>

<div align="center">
  <img src="docs/_static/figures/big-flow.svg" width="920" alt="Whole FlowEdge flow: convert, load arena, Mamba+flow or LeRobot encoder+DP, period, motors">
</div>

<div align="center">
  <img src="docs/_static/figures/relay.svg" width="920" alt="Optional Relay: sensor or VLA to shared-memory rings to Core to controller, same miss contract">
</div>

<div align="center">
  <img src="docs/_static/figures/workflow.svg" width="920" alt="convert, load arena, sample flow or DDIM, period, act">
</div>

## Python

```bash
python -m pip install .
python examples/core/flow_sample.py models/mamba_flow.safetensors
```

`Engine.sample(..., method="euler"|"heun"|"rk4")`. Wheels: [v0.1.1](https://github.com/elprofesoriqo/FlowEdge/releases/tag/v0.1.1) (not PyPI).

<div align="center">
  <img src="docs/_static/figures/status.svg" width="920" alt="Implementation status: backends, backbones, heads, models, precisions">
</div>

```bash
python -m flowedge_dev --help
```

Local merge gate: `scripts/verify_all.sh` or `scripts/verify_local.ps1`. GitHub Actions is compile + `ctest`.
