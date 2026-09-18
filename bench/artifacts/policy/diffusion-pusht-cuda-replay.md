# LeRobot / FlowEdge edge benchmark

> These are measurements of one checkpoint on one declared host, not universal deployment guarantees.

## Run identity

| Field | Value |
|---|---|
| Captured | `2026-09-18T12:52:38.795231+00:00` |
| Model | `lerobot/diffusion_pusht` @ `84a7c23178445c6bbf7e1a884ff497017910f653` |
| Model SHA-256 | `995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8` |
| Processor | `lerobot/diffusion_pusht` @ `84a7c23178445c6bbf7e1a884ff497017910f653` |
| Processor stats | `{"observation.image":{"mean":[[[0.48500001430511475]],[[0.4560000002384186]],[[0.4059999883174896]]],"std":[[[0.2290000021457672]],[[0.2240000069141388]],[[0.22499999403953552]]]},"observation.state":{"max":[496.14617919921875,510.9578857421875],"min":[13.45642375946045,32.93829345703125]}}` |
| Observation contract | `2dde368a238be30a27b8a7c243b724c51d7c744946f28e439c2afc625e2904fe`; 2 steps |
| Action contract | 2 dims x 8 steps; dataset |
| Run shape | batch 1; 10 inference steps |
| Host | `Jankowski`; Linux-5.15.153.1-microsoft-standard-WSL2-x86_64-with-glibc2.39; x86_64; 1 threads; CUDA `NVIDIA GeForce GTX 1650` |
| Build | `GCC 13.3.0`; Release |
| Comparison | FlowEdge native runtime vs LeRobot policy executed with PyTorch |
| Reference framework | `PyTorch` |
| Scope | matched observation encoder, history, noise, and DDIM schedule on CUDA; FlowEdge engine is freed before the PyTorch U-Net is loaded; not TensorRT/ONNX; not Jetson/ARM |
| Max abs error | 7.62939e-05 (atol=0.001, rtol=0.0001) |

## Measurements

Encoder and policy are reported separately. End-to-end includes both and is the number to use for control-loop budgeting. The reference row is the same LeRobot policy executed through PyTorch.

| Backend | Startup | Encoder p50 / p95 / p99 | Policy p50 / p95 / p99 | E2E p50 / p95 / p99 | Throughput | RSS | Setup / hot allocations |
|---|---:|---:|---:|---:|---:|---:|---:|
| flowedge | 31494.321 ms | 6.001 ms / 7.447 ms / 7.492 ms | 153.491 ms / 155.381 ms / 155.475 ms | 159.454 ms / 162.131 ms / 162.520 ms | 6.258 Hz | 3144.398 MiB | not measured / not measured |
| lerobot | 32203.443 ms | 5.493 ms / 13.240 ms / 14.507 ms | 361.247 ms / 370.623 ms / 370.740 ms | 367.935 ms / 377.494 ms / 377.899 ms | 2.717 Hz | 3144.398 MiB | not measured / not measured |

## Relative comparison

| Metric | FlowEdge / LeRobot |
|---|---:|
| Policy p50 | 0.425x |
| End-to-end p50 | 0.433x |
| Throughput | 2.303x |

## Commands

```text
FlowEdge: /home/igor/fe-cuda-venv/bin/python -m flowedge_lerobot.benchmark models/diffusion_pusht.flowedge.safetensors --source models/diffusion_pusht --revision 84a7c23178445c6bbf7e1a884ff497017910f653 --model-id lerobot/diffusion_pusht --steps 10 --iterations 10 --warmup 2 --threads 1 --device cuda --seed 7 --output bench/artifacts/policy/diffusion-pusht-cuda-replay.json --observations bench/artifacts/policy/diffusion-pusht-cpu-replay.observations.npz
LeRobot:  /home/igor/fe-cuda-venv/bin/python -m flowedge_lerobot.benchmark models/diffusion_pusht.flowedge.safetensors --source models/diffusion_pusht --revision 84a7c23178445c6bbf7e1a884ff497017910f653 --model-id lerobot/diffusion_pusht --steps 10 --iterations 10 --warmup 2 --threads 1 --device cuda --seed 7 --output bench/artifacts/policy/diffusion-pusht-cuda-replay.json --observations bench/artifacts/policy/diffusion-pusht-cpu-replay.observations.npz
```

## Limits

- Repeat on the target CPU, OS, compiler, thread count, power mode, and checkpoint before making a deployment decision.
- RSS includes process/runtime state and is not a per-inference allocation measurement.
- The zero-allocation contract applies to supported native paths; unknown full-policy counts remain unmeasured.
- CUDA full-policy replay of one fixed PushT observation history; not closed-loop task success.
- End-to-end includes preprocessing, encoder, history and sampling; excludes camera capture, IPC and robot delivery.
- RSS is peak process high-water memory; CUDA frees the FlowEdge engine before loading the PyTorch U-Net, so both U-Nets are not device-resident together.
- Allocations are unmeasured; no zero-allocation claim applies to this Python visual pipeline.
- Startup excludes interpreter/import time and uses the existing filesystem cache.
- Small iteration counts do not establish stable p99 estimates.
- CUDA device name is the headline; host threads record leftover CPU work, not a GPU fair-compare.
- Not Jetson/ARM evidence. Not TensorRT/ONNX.
- A 4GB GPU cannot hold the FlowEdge resident DDIM and the PyTorch U-Net at once.
