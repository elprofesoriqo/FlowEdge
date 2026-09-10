# Python API

```bash
python -m pip install .
```

## Fast path

```python
import numpy as np
import flowedge

engine = flowedge.Engine("models/mamba_flow.safetensors", threads=0)
tokens = np.array([1, 2, 3, 4], dtype=np.int32)
noise = np.zeros(engine.action_dim, dtype=np.float32)
action = np.empty(engine.action_dim, dtype=np.float32)
engine.sample_into(tokens, noise, action, steps=10, method="euler")
```

| API family | Convenience form | Caller-owned form |
|---|---|---|
| Backbone | `run(tokens)`, `step(token)` | `run_into(tokens, hidden)`, `step_into(token, out)` |
| Flow head | `sample(prefix, noise, ...)` | `sample_into(prefix, noise, out, ...)` |
| External encoder | — | `sample_condition(condition, noise, out, ...)` |
| Diffusion | `sample_diffusion(condition, noise, ...)` | `sample_diffusion_into(condition, noise, out, ...)` |
| State | `decode_state()` / `restore_decode_state(bytes)` | Caller owns snapshot bytes |

Convenience calls normalize array dtype/layout and allocate returned arrays. Real-time loops should
use C-contiguous `int32`/`float32` inputs and caller-owned outputs. C++ execution releases the GIL.

## Engine configuration

| Option | Meaning |
|---|---|
| `threads=None` | `FLOWEDGE_THREADS`, then automatic default |
| `threads=0` | Caller-thread-only execution |
| `threads=1..8` | Fixed worker count |

## External encoder and resumable solve

```python
condition = encoder(observation).astype(np.float32, copy=False)
noise = np.zeros(engine.action_dim, dtype=np.float32)
action = np.empty(engine.action_dim, dtype=np.float32)
engine.sample_condition(condition, noise, action, 8, "heun")

engine.flow_begin(condition, noise, 8, "heun", generation=42,
                  timestamp_ns=observation_time, deadline_ns=control_deadline)
while engine.flow_advance(action, 1):
    pass
```

| Rule | Contract |
|---|---|
| Ownership | One engine owns one mutable stream or active solve |
| Cancellation | `cancel_before(generation)` invalidates older work at a solver boundary |
| Publishing | Publish a flow output only when `flow_advance` reports zero remaining steps |
| Identity | Condition dimensions and model digest must match |

## Streaming snapshots

`decode_state()` returns a versioned, checksummed, little-endian snapshot. Restore rejects a
different model digest, architecture, precision, dimensions, truncation, or corruption before
changing state.

## Diffusion Policy

```python
engine = flowedge.Engine("models/diffusion_pusht.flowedge.safetensors")
condition = observation_encoder(history).astype(np.float32, copy=False)
noise = np.random.default_rng(7).standard_normal(
    (engine.action_horizon, engine.action_dim), dtype=np.float32
)
actions = np.empty_like(noise)
engine.sample_diffusion_into(condition, noise, actions, steps=10, scheduler="ddim")
```

| Metadata | Meaning |
|---|---|
| `action_horizon` | Full predicted horizon |
| `action_dim` | Values per action |
| `diffusion_metadata` | Horizon, action slice, train timesteps, clipping policy |
| LeRobot slice | `actions[observation_steps - 1 : observation_steps - 1 + action_steps]` |

The LeRobot companion adapter exposes the same allocation-conscious pattern through
`predict_action_chunk_into` and `select_action_into`; reuse those buffers in a control loop.

`e.diffusion_metadata` exposes `horizon`, `action_steps`, `observation_steps`,
`train_timesteps`, and the clipping policy. LeRobot normally executes
`actions[observation_steps - 1: observation_steps - 1 + action_steps]` from the
returned horizon; the controller owns that slicing decision.
