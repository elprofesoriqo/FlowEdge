# LeRobot deployment adapter

LeRobot trains the policy, runs cameras, and talks to the robot. FlowEdge is the native action head behind that plugin: converted `diffusion_pusht` first, then a cached SmolVLA expert. The problem this solves is the last milliseconds — a fixed U-Net, a period, and an explicit miss policy — without putting e-stop or joint limits inside Core.

```{image} ../_static/figures/purpose.svg
:alt: Hardware, FlowEdge, LeRobot
:class: fe-fig
```

```{image} ../_static/figures/deadline.svg
:alt: 10 ms period vs sample latency, on-miss hold
:class: fe-fig
```

## Boundary

| LeRobot | FlowEdge |
|---|---|
| Cameras, RGB/state encoder, normalization, history | Fixed-memory U-Net / expert |
| Robot driver, joint limits, e-stop | `--on-miss hold\|drop\|raise` |
| Training | Converted `.safetensors` |

Action slice: `actions[observation_steps - 1 : observation_steps - 1 + action_steps]`.
`input_mode="visual"` uses the source encoder. A flattened condition remains valid. [Policy evaluation](policy-evaluation).

## Install

```bash
python -m pip install -e integrations/lerobot
python -m unittest discover -s integrations/lerobot/tests
```

```python
from flowedge_lerobot import FlowEdgeDiffusionPolicy
policy = FlowEdgeDiffusionPolicy.from_checkpoint(
    "models/diffusion_pusht.flowedge.safetensors", threads=4
)
```

## Period loop (sim, Jetson, SO-100)

```bash
python -m flowedge_dev pipeline rollout models/diffusion_pusht.flowedge.safetensors \
  --steps 20 --threads 4 --period-ms 10 --on-miss hold
```

CUDA Core (`FLOWEDGE_BACKEND=cuda`), same miss contract:

```bash
python -m flowedge_dev pipeline rollout models/diffusion_pusht.flowedge.safetensors \
  --steps 20 --threads 1 --period-ms 10 --on-miss hold --device cuda
```

`input_mode=visual` keeps the source RGB encoder in LeRobot/PyTorch (`device=cuda`).
GTX 1650, 10 ms hold: 20 / 20 misses. Not a Jetson/ARM claim.

| `--on-miss` | On overrun |
|---|---|
| `hold` | Repeat last sent action; zeros before the first on-time send |
| `drop` | Skip `send_action` |
| `raise` | `DeadlineMissed` after `stop` |

Jetson: `scripts/edge_dp_rollout.sh`. Attach JSON to issue #70. SO-100: implement `reset` / `observe` / `send_action` / `stop` and pass `encode_condition` into `run_rollout`. CI uses a fake robot. ARM64 CI is architecture validation, not a board claim.

`--policy.type=flowedge` is DP. `--policy.type=flowedge_smolvla` is the native expert plus a LeRobot VLM cache — not a native VLM. Tenstorrent is a separate backend, not this plugin.
