# LeRobot deployment adapter

FlowEdge's LeRobot integration is a companion adapter, not a change to the universal Core or
Relay runtime. The first supported path is the converted `lerobot/diffusion_pusht` checkpoint.

## Boundary

LeRobot remains responsible for observation capture, image/state encoding, feature normalization,
robot limits, and emergency-stop behavior. FlowEdge receives the flattened condition, runs the
fixed-memory Diffusion Policy head, un-normalizes actions using the converted checkpoint metadata,
and returns the LeRobot action slice:

```text
actions[observation_steps - 1 : observation_steps - 1 + action_steps]
```

This boundary keeps LeRobot-specific behavior outside `src/core` and allows the same FlowEdge
runtime to serve other policy/tooling integrations.

## Install and test the companion adapter

From a FlowEdge checkout with the main Python package installed:

```bash
python -m pip install -e integrations/lerobot
python -m unittest discover -s integrations/lerobot/tests
```

Use the adapter with a converted checkpoint:

```python
import numpy as np
from flowedge_lerobot import FlowEdgeDiffusionPolicy

policy = FlowEdgeDiffusionPolicy.from_checkpoint(
    "models/diffusion_pusht.flowedge.safetensors", threads=4
)
condition = np.zeros(policy.metadata.condition_dim, dtype=np.float32)
noise = np.zeros((policy.metadata.horizon, policy.metadata.action_dim), dtype=np.float32)
chunk = policy.predict_action_chunk(condition, noise, steps=10)
```

The companion adapter is deployment-only; training remains in LeRobot. A complete
`lerobot-rollout` plugin, processor parity checks, and SO-100/SO-101 demo are tracked in
[issue #65](https://github.com/elprofesoriqo/FlowEdge/issues/65) and its subissues.

Tenstorrent support is intentionally not part of this integration slice. It remains a separate
backend effort so this adapter does not couple the universal runtime to one accelerator.
