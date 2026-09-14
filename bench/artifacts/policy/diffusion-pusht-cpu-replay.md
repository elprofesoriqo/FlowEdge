# LeRobot / FlowEdge edge benchmark

> These are measurements of one checkpoint on one declared host, not universal deployment guarantees.

## Run identity

| Field | Value |
|---|---|
| Captured | `2026-09-13T14:00:38.402081+00:00` |
| Model | `lerobot/diffusion_pusht` @ `84a7c23178445c6bbf7e1a884ff497017910f653` |
| Model SHA-256 | `995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8` |
| Processor | `lerobot/diffusion_pusht` @ `84a7c23178445c6bbf7e1a884ff497017910f653` |
| Processor stats | `{"observation.image":{"mean":[[[0.48500001430511475]],[[0.4560000002384186]],[[0.4059999883174896]]],"std":[[[0.2290000021457672]],[[0.2240000069141388]],[[0.22499999403953552]]]},"observation.state":{"max":[496.14617919921875,510.9578857421875],"min":[13.45642375946045,32.93829345703125]}}` |
| Observation contract | `2dde368a238be30a27b8a7c243b724c51d7c744946f28e439c2afc625e2904fe`; 2 steps |
| Action contract | 2 dims x 8 steps; dataset |
| Run shape | batch 1; 10 inference steps |
| Host | `Jankowski`; Windows-10-10.0.26200-SP0; Intel64 Family 6 Model 158 Stepping 13, GenuineIntel; 1 threads |
| Build | `Clang 23.1.0 (https://github.com/llvm/llvm-project ea7d852a70e8bdfaf601d6626a760f9771b2c4b4)`; Release |

## Measurements

Encoder and policy are reported separately. End-to-end includes both and is the number to use for control-loop budgeting.

| Backend | Startup | Encoder p50 / p95 / p99 | Policy p50 / p95 / p99 | E2E p50 / p95 / p99 | Throughput | RSS | Setup / hot allocations |
|---|---:|---:|---:|---:|---:|---:|---:|
| flowedge | 998.002 ms | 101.249 ms / 130.462 ms / 138.550 ms | 25848.085 ms / 26390.380 ms / 26541.522 ms | 25966.772 ms / 26488.653 ms / 26643.189 ms | 0.046 Hz | 3293.434 MiB | not measured / not measured |
| lerobot | 1145.402 ms | 100.764 ms / 115.043 ms / 128.573 ms | 3021.965 ms / 3061.190 ms / 3066.505 ms | 3120.536 ms / 3164.034 ms / 3171.596 ms | 0.382 Hz | 3293.434 MiB | not measured / not measured |

## Relative comparison

| Metric | FlowEdge / LeRobot |
|---|---:|
| Policy p50 | 8.553x |
| End-to-end p50 | 8.321x |
| Throughput | 0.121x |

## Commands

```text
FlowEdge: C:\Users\igorj\Desktop\FlowEdge\venv\Scripts\python.exe -m flowedge_lerobot.benchmark models/diffusion_pusht.flowedge.safetensors --source models/diffusion_pusht --revision 84a7c23178445c6bbf7e1a884ff497017910f653 --steps 10 --iterations 20 --warmup 2 --threads 1 --output bench/artifacts/diffusion-pusht-cpu-replay.json
LeRobot:  C:\Users\igorj\Desktop\FlowEdge\venv\Scripts\python.exe -m flowedge_lerobot.benchmark models/diffusion_pusht.flowedge.safetensors --source models/diffusion_pusht --revision 84a7c23178445c6bbf7e1a884ff497017910f653 --steps 10 --iterations 20 --warmup 2 --threads 1 --output bench/artifacts/diffusion-pusht-cpu-replay.json
```

## Limits

- Repeat on the target CPU, OS, compiler, thread count, power mode, and checkpoint before making a deployment decision.
- RSS includes process/runtime state and is not a per-inference allocation measurement.
- Setup allocations may be non-zero; the FlowEdge hot-path count is expected to remain zero after setup.
- CPU full-policy replay of one fixed PushT observation history; not closed-loop task success.
- End-to-end includes preprocessing, encoder, history and sampling; excludes camera capture, IPC and robot delivery.
- RSS is shared process high-water memory with both implementations resident, not per-backend memory.
- Allocations are unmeasured; no zero-allocation claim applies to this Python visual pipeline.
- Startup excludes interpreter/import time and uses the existing filesystem cache.
- Small iteration counts do not establish stable p99 estimates.
