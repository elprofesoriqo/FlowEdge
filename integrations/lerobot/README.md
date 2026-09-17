# FlowEdge + LeRobot

![Hardware, FlowEdge, LeRobot](../../docs/_static/figures/purpose.svg)

Out-of-tree `lerobot_policy_flowedge` plugin. Training, cameras, limits, and e-stop stay in LeRobot. FlowEdge runs the converted action path.

## Scope

- Converted `lerobot/diffusion_pusht` (`--policy.type=flowedge`)
- Visual observations via the source RGB encoder, or a flattened condition
- `--on-miss hold|drop|raise` on a control period
- Cached SmolVLA expert (`--policy.type=flowedge_smolvla`): LeRobot VLM cache, native Euler expert — not a native VLM

Action slice: `actions[observation_steps - 1 : observation_steps - 1 + action_steps]`.

## Install

```bash
python -m pip install -e integrations/lerobot
python -m unittest discover -s integrations/lerobot/tests
```

## Hardware and sim

Fake robot (CI and CPU):

```bash
python -m flowedge_dev pipeline rollout models/diffusion_pusht.flowedge.safetensors \
  --steps 20 --threads 4 --period-ms 10 --on-miss hold
```

Jetson / ARM: `scripts/edge_dp_rollout.sh <converted.safetensors>` and attach JSON to issue #70.

SO-100/SO-101: wrap `reset` / `observe` / `send_action` / `stop` and pass `encode_condition` into `run_rollout`. The plugin does not import a robot driver.

```python
from flowedge_lerobot import FlowEdgeDiffusionPolicy, run_rollout

policy = FlowEdgeDiffusionPolicy.from_checkpoint("models/diffusion_pusht.flowedge.safetensors")
result = run_rollout(policy, robot, encode_condition=processor, steps=100, seed=7)
```

## SmolVLA expert

VLM stays in LeRobot. Hybrid parity / timing:

```bash
python -m flowedge_dev verify smolvla \
  models/smolvla_base/model.safetensors models/smolvla_base \
  bench/artifacts/smolvla/eslab-frame-000000.capture.npz \
  --module-path build-research
python -m flowedge_dev bench hybrid ...   # stage p50/p95/p99
```

Not native SmolVLA. Not a control-loop claim. Details: [policy evaluation](../../docs/guides/policy-evaluation.md).
