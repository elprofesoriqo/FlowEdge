# FlowEdge + LeRobot adapter

This companion package is a LeRobot-discoverable `lerobot_policy_flowedge` deployment plugin. It deliberately
does not change `src/core` or `src/relay`, and it does not replace LeRobot's observation encoder.

## Current scope

- Converted `lerobot/diffusion_pusht` checkpoints.
- Encoded conditions, or source visual observations with the supported RGB encoder and statistics.
- FlowEdge DDIM/DDPM inference and LeRobot action-horizon slicing.
- Experimental SmolVLA cached-expert chunks: LeRobot/source code produces the VLM cache and
  FlowEdge executes the seeded native action expert.
- Deployment-only Python adapter; training remains in LeRobot.

The adapter follows the same action contract documented by FlowEdge's Diffusion Policy guide:
LeRobot consumes `actions[observation_steps - 1: observation_steps - 1 + action_steps]` from the
full denoised horizon. The visual companion path owns normalization, image encoding, and
observation history outside native Core. Robot limits and emergency-stop behavior remain application-owned.

The plugin consumes chunks across `select_action` calls, uses seeded Gaussian noise, and clears
both history and queued actions on reset. See [policy evaluation](../../docs/guides/policy-evaluation.md)
for `input_mode`, source checkpoint setup, matched replay, PushT, and periodic delivery.

## Experimental SmolVLA cache boundary

`FlowEdgeSmolVLACachedExpert` and `LeRobotSmolVLACacheProvider` are programmatic
hybrid adapters, not a registered `--policy.type` and not native full SmolVLA.
Install the optional source stack with `pip install -e 'integrations/lerobot[smolvla]'`.
The provider follows an instantiated source-compatible LeRobot VLM to produce
its batch-one, RoPE-applied K/V cache; the adapter then runs the
native FlowEdge Euler action expert, truncates padded action coordinates, and
queues the requested action chunk. Preprocessing, VLM execution, action
postprocessing, and cache-transfer timing remain outside this boundary.

The real hybrid parity gate exercises this boundary with a recorded LeRobot
frame and compares the complete `50 x 6` action chunk against the source
`SmolVLAPolicy`:

```bash
python tools/verification/verify_smolvla_hybrid.py \
  models/smolvla_base/model.safetensors models/smolvla_base \
  bench/artifacts/smolvla/eslab-frame-000000.capture.npz \
  --module-path build-research \
  --output bench/artifacts/smolvla/eslab-frame-000000.hybrid-action.json
```

This is source-pipeline plus native-expert evidence, not native VLM or full
native SmolVLA deployment evidence. It also does not measure latency or
control quality.

For stage timing on the same real capture, use the separate benchmark entry
point. It reports p50/p95/p99 for source PyTorch action generation, source VLM
prefix/cache production, native action-expert execution, and the combined
hybrid path:

```bash
python tools/benchmark/run_smolvla_hybrid_report.py \
  models/smolvla_base/model.safetensors models/smolvla_base \
  bench/artifacts/smolvla/eslab-frame-000000.capture.npz \
  --module-path build-research --iterations 20 --warmup 5 \
  --output bench/artifacts/smolvla/eslab-frame-000000.hybrid-report.json
```

These are prepared-capture CPU measurements, not full control-loop latency or
real-time suitability evidence.

## Development install

From a checkout with the main `flowedge` package installed:

```bash
python -m pip install -e integrations/lerobot
python -m unittest discover -s integrations/lerobot/tests
```

The plugin is intentionally independent from LeRobot's release cycle. It registers `--policy.type=flowedge`
and accepts a converted checkpoint through `checkpoint_path`; it is inference-only, so training remains in
LeRobot. The
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

The command prints JSON telemetry (`steps`, `elapsed_ms`, `p50_ms`, `p99_ms`, `max_ms`,
`missed_deadlines`, `startup_ms`, `peak_rss_bytes`, platform, and machine) and never imports a concrete robot driver. `peak_rss_bytes` is process high-water RSS and can be `null` when the host cannot expose it. Add `--period-ms` to count steps
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
