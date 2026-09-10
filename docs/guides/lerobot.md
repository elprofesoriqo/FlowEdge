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

The companion adapter is deployment-only; training remains in LeRobot. Processor parity checks,
hardware smoke testing, and partner validation are tracked in [issue #65](https://github.com/elprofesoriqo/FlowEdge/issues/65)
and its subissues.

## SO-100/SO-101 rollout seam

The companion package includes `run_rollout`, a bounded single-action loop that accepts a small
robot shim (`reset`, `observe`, `send_action`, and `stop`) plus an observation encoder. It is
suitable for wiring the adapter into SO-100/SO-101 applications without adding a robot driver or
preprocessing dependency to FlowEdge. The loop always calls `stop`, including when an encoder or
inference call raises, and uses a bounded step count for repeatable dry runs.

See `integrations/lerobot/README.md` for the minimal wrapper example. Hardware validation remains
a follow-up smoke test on a real LeRobot robot; repository tests use a fake robot.

## Simulator smoke test

The companion package exposes a bounded simulator command for checking a converted checkpoint
before wiring a robot driver:

```bash
python -m pip install -e integrations/lerobot
flowedge-lerobot-rollout models/diffusion_pusht.flowedge.safetensors --steps 10
```

It prints JSON completion and timing telemetry. Pass `--period-ms 10` to count missed 10 ms
control-loop periods. Observation encoding, limits, and emergency-stop behavior stay in the
caller-owned robot adapter.

Tenstorrent support is intentionally not part of this integration slice. It remains a separate
backend effort so this adapter does not couple the universal runtime to one accelerator.

## ARM64/NEON validation

Every CI run includes an `ubuntu-24.04-arm` job. It builds the Release Core and Relay targets with
`FLOWEDGE_BACKEND=cpu`, runs the native tests and adapter contract tests, then publishes a profile
artifact containing compiler, CPU, latency, and allocation metadata. This is architecture
validation, not a claim for a specific Jetson, RDK, or robot; hardware pilots still need their
own processor and safety checks.
