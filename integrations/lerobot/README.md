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

The companion package is intentionally independent from LeRobot's release cycle. The next issue
adds a complete `lerobot-rollout` plugin and hardware demo after parity tests are accepted.

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
