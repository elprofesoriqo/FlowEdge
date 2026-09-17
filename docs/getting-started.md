# Getting Started

FlowEdge is the deploy runtime, not the trainer. Convert a checkpoint, load it
once, sample **flow matching** or **Diffusion Policy**. Optional Relay sits
between processes when perception and control are split.

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

```{image} _static/figures/policies.svg
:alt: Flow matching ODE versus Diffusion Policy DDIM
:class: fe-fig
```

```{image} _static/figures/workflow.svg
:alt: convert, load arena, sample flow or DDIM, period, act
:class: fe-fig
```

| Goal | Continue at |
|---|---|
| Included flow-matching policy | [Sample an action](sample-an-action) |
| Diffusion Policy on Jetson / SO-100 / sim | [LeRobot adapter](guides/lerobot) |
| Your encoder, our flow or DP head | [External encoder](use-an-external-encoder) |
| Local IPC | [Relay](relay-end-to-end-demo) |
| Python / NumPy | [Python](python-quickstart) |
| Cached SmolVLA expert (not a product head) | [Transformer / SmolVLA](guides/transformer-backbone) |
| Flow matching vs Diffusion Policy vs PyTorch | [Performance](performance) |

GPT-2 conversion is a kernel incubator, not a product path.

## Build

CMake 3.21+ and C++23. CI uses LLVM/Clang 23; GCC 13+ is supported.

```bash
git clone https://github.com/elprofesoriqo/FlowEdge.git
cd FlowEdge
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

## Get a model

```bash
mkdir -p models
wget -qO models/mamba_flow.safetensors \
  "https://huggingface.co/ReForceMind/mamba_flow/resolve/main/mamba_flow.safetensors"
```

LeRobot: `python -m flowedge_dev pipeline convert` ([converter](guides/converter)).

(sample-an-action)=
## Sample an action

```bash
./build/flow_sample models/mamba_flow.safetensors euler 10
```

Solver is `euler`, `heun`, or `rk4`. Last argument is NFE (how many velocity-net evaluations). That is the cost of flow matching: a short ODE, not a long denoiser.

Converted Diffusion Policy — plugin owns the LeRobot processor; Core owns the U-Net:

```bash
python -m pip install -e integrations/lerobot
python -m flowedge_dev pipeline rollout models/diffusion_pusht.flowedge.safetensors \
  --steps 10 --period-ms 10 --on-miss hold
```

`--on-miss`: `hold` last sent action, `drop` the send, or `raise`. Limits and e-stop stay in the robot adapter. Jetson: `scripts/edge_dp_rollout.sh`.

(use-an-external-encoder)=
## Use an external encoder

Keep ResNet / VLM / your own encoder where it already runs. Pass the condition vector in; FlowEdge only integrates the head.

```bash
./build/external_flow_sample models/mamba_flow.safetensors
./build/streaming_snapshot models/mamba_flow.safetensors
```

Caller owns condition, noise, and output. Engine owns solver scratch.

(relay-end-to-end-demo)=
## Relay

Optional local IPC when perception and control are separate processes. Matched DP replay does not start Relay.

```bash
cmake -S . -B build-relay -DCMAKE_BUILD_TYPE=Release -DFLOWEDGE_RELAY=ON
cmake --build build-relay -j
FLOWEDGE_BUILD_DIR="$PWD/build-relay" ./scripts/relay_demo.sh models/mamba_flow.safetensors
```

[Relay quickstart](guides/relay-quickstart).

(python-quickstart)=
## Python

```bash
pip install .
```

```python
import numpy as np, flowedge
e = flowedge.Engine("models/mamba_flow.safetensors")
a = e.sample(prefix=np.array([1, 2, 3, 4], dtype=np.int32),
             noise=np.random.randn(e.action_dim).astype("float32"),
             steps=10, method="euler")
```

Load may allocate once. Supported hot paths allocate nothing after init. One engine, one stream. Backend is CPU.
