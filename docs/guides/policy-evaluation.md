# Policy evaluation

The companion package separates integration smoke tests, matched full-policy replay,
and closed-loop PushT evaluation. The last two use the original RGB encoder,
normalization statistics, observation history, and trained Diffusion Policy weights.
The first public fixture is `lerobot/diffusion_pusht` revision
`84a7c23178445c6bbf7e1a884ff497017910f653`.

## Setup

```bash
python -m pip install .
python -m pip install -e 'integrations/lerobot[evaluation]'
hf download lerobot/diffusion_pusht --revision 84a7c23178445c6bbf7e1a884ff497017910f653 \
  --include config.json model.safetensors --local-dir models/diffusion_pusht
python convert/convert.py models/diffusion_pusht models/diffusion_pusht.flowedge.safetensors \
  --arch diffusion --dtype f32
```

The pinned gym-pusht release needs Pymunk 6; its upstream dependency also permits
incompatible Pymunk 7. The evaluation extra constrains that dependency.

## Plugin semantics

`FlowEdgeConfig.input_mode="encoded"` accepts one already-normalized, complete
condition vector in `observation.state`. It rejects raw dataset statistics and images.
`input_mode="visual"` also requires `source_checkpoint_path` pointing at the original
model directory. Its processor loads source statistics or validates explicit
`dataset_stats`; its encoder maintains the source observation history. Use source
feature definitions in the plugin config. Policy methods accept processor outputs.
Converted checkpoints bind the source model and config hashes. Reconvert older artifacts
before visual deployment; a mismatched encoder/checkpoint pair is rejected at setup.

`select_action` updates observations every tick, consumes the current chunk, then
resamples on depletion. `action_steps=None` adopts the converted executable horizon;
a shorter execution horizon is allowed. `reset` clears history/actions and restarts
the configured seed. `predict_action_chunk` predicts a fresh chunk and invalidates
queued actions. Optional full-horizon `noise` overrides sampling at the next refill.

Native output is already in dataset action units; the output processor must not
unnormalize it again. Encoding uses PyTorch on CPU, outside Core's allocation contract.
Unsupported source layouts fail during setup; this is not a universal LeRobot adapter.

## Matched replay

```bash
python tools/benchmark/run_policy_report.py models/diffusion_pusht.flowedge.safetensors \
  --source models/diffusion_pusht --revision 84a7c23178445c6bbf7e1a884ff497017910f653 \
  --steps 10 --iterations 100 --warmup 5 --threads 1 \
  --output bench/artifacts/policy/diffusion-pusht-report.json
```

The runner alternates execution order and checks actions against the upstream LeRobot
PyTorch U-Net and diffusers DDIM scheduler. It saves raw timings, source/converted/config hashes,
processor statistics, versions, and replay observations. End-to-end means preprocessing
through executable action chunk; capture, IPC, and robot delivery are excluded. RSS is
shared process high-water memory, not per-backend usage. Uninstrumented allocations are
`null` with a reason, never an invented zero. One fixed history is a replay workload,
not task quality. Small sample counts do not establish stable tail-latency estimates.

## Closed-loop tasks

```bash
python -m flowedge_lerobot.evaluate models/diffusion_pusht.flowedge.safetensors \
  --source models/diffusion_pusht --backend flowedge --episodes 5 --max-steps 300 \
  --steps 10 --threads 1 --output bench-results/pusht-flowedge.json
```

Repeat with `--backend lerobot` and identical seeds/settings. Reports include seeds,
reward, success, steps, and hashes. This mode advances simulated time after an action
becomes available; wall-clock latency does not penalize its score.

Add `--period-ms 100` for periodic asynchronous delivery. Startup bootstraps the first
chunk before timing; later chunks discard actions whose observation-relative ticks have
elapsed. Reports include underruns, delivery deadline misses, absolute inter-delivery
jitter, and observation age. PushT's explicit fallback holds its current agent position.
Real robots must supply appropriate fallback and stop behavior. The Python runner can
allocate and is subject to OS and GIL scheduling.

`flowedge-lerobot-rollout` remains a synthetic integration smoke test. Its
`missed_deadlines` counts per-call overruns, not periodic deadline misses.
