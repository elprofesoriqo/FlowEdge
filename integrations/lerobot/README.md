# FlowEdge + LeRobot adapter

This companion package is the first LeRobot integration boundary for FlowEdge. It deliberately
does not change `src/core` or `src/relay`, and it does not replace LeRobot's observation encoder.

## Current scope

- Converted `lerobot/diffusion_pusht` checkpoints.
- Flattened, already-preprocessed observation conditions.
- FlowEdge DDIM/DDPM inference and LeRobot action-horizon slicing.
- Deployment-only Python adapter; training remains in LeRobot.

The adapter follows the same action contract documented by FlowEdge's Diffusion Policy guide:
LeRobot consumes `actions[observation_steps - 1: observation_steps - 1 + action_steps]` from the
full denoised horizon. Feature normalization, image encoding, robot limits, and emergency-stop
behavior remain outside FlowEdge.

## Development install

From a checkout with the main `flowedge` package installed:

```bash
python -m pip install -e integrations/lerobot
python -m unittest discover -s integrations/lerobot/tests
```

The companion package is intentionally independent from LeRobot's release cycle. The
`flowedge-lerobot-rollout` command provides a bounded simulator smoke test; hardware validation
and processor parity remain tracked in issue #65 and its subissues.

## SO-100/SO-101 rollout shim

`flowedge_lerobot.run_rollout` provides a bounded single-action loop without importing a concrete
LeRobot robot class. Wrap the robot's `reset`, observation, action, and stop methods, and keep the
LeRobot camera/state processor in `encode_condition`:

```python
from flowedge_lerobot import FlowEdgeDiffusionPolicy, run_rollout

policy = FlowEdgeDiffusionPolicy.from_checkpoint("models/diffusion_pusht.flowedge.safetensors")
result = run_rollout(
    policy,
    robot,
    encode_condition=lambda observation: processor(observation),
    steps=100,
    seed=7,
)
assert result.stopped
```

For a control loop that owns its buffers, use `predict_action_chunk_into` or
`select_action_into`. The adapter reuses one full-horizon workspace and the rollout shim reuses
its noise and action arrays, avoiding per-step NumPy allocations. Calls are single-threaded per
policy instance because that workspace is intentionally shared; `send_action` must consume the
borrowed action before the next loop iteration.

This is the integration seam for SO-100/SO-101 applications; robot drivers, feature processors,
joint limits, and hardware emergency-stop behavior remain application-owned. The included tests
use a fake robot so the loop can be checked without hardware.

## Simulator smoke test

Install the companion package, then run a bounded rollout against a converted checkpoint:

```bash
python -m pip install -e integrations/lerobot
flowedge-lerobot-rollout models/diffusion_pusht.flowedge.safetensors --steps 10
```

The command prints JSON telemetry (`steps`, `elapsed_ms`, `p50_ms`, `p99_ms`, `max_ms`, and
`missed_deadlines`) and never imports a concrete robot driver. Add `--period-ms` to count steps
over a control-loop deadline. Replace `run_simulator` with a `RolloutRobot` adapter when
connecting SO-100/SO-101 hardware.

## Example

```python
import numpy as np
from flowedge_lerobot import FlowEdgeDiffusionPolicy

policy = FlowEdgeDiffusionPolicy.from_checkpoint(
    "models/diffusion_pusht.flowedge.safetensors", threads=4
)
condition = np.zeros(policy.metadata.condition_dim, dtype=np.float32)
noise = np.zeros((policy.metadata.horizon, policy.metadata.action_dim), dtype=np.float32)
action_chunk = policy.predict_action_chunk(condition, noise, steps=10)
```

This is a deployment adapter, not a claim that every LeRobot policy or hardware target is
supported. See issue #65 and its subissues for the rollout plugin, ARM validation, and partner
pilot work.
