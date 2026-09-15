# Benchmark map

FlowEdge has separate evidence for kernel speed, runtime predictability, and
complete policy deployment. Do not use a component number as a control-loop
claim.

```{mermaid}
flowchart LR
  K[Kernel microbench] --> S[Throughput / CPU cost]
  R[Relay + queue bench] --> T[Tail latency / allocations]
  P[LeRobot replay] --> C[PyTorch parity + p50/p95/p99]
  E[PushT episodes] --> Q[Task outcome / deadline evidence]
  S & T & C & Q --> A[Versioned artifact + host parameters]
```

| Suite | Location | What it measures | Default evidence |
|---|---|---|---|
| Kernels | `bench/kernels/` | Dense kernels and threaded matmul | Google Benchmark JSON |
| Runtime | `bench/runtime/` | Relay admission, queues, worker pools, delivery, streaming, deadline profile | p50/p95/p99, cancellation, hot allocations |
| Transformer | `tools/verification/verify_transformer_reference.py` | Real Hugging Face checkpoint against native full-prefix and streaming execution | max error and parity |
| Diffusion policy | `tools/benchmark/run_policy_report.py` | Same observations, processor, noise, and DDIM schedule through FlowEdge and LeRobot/PyTorch | encoder/policy/end-to-end p50/p95/p99, throughput, RSS, action error |
| Closed loop | `flowedge_lerobot.evaluate` | Bounded PushT episodes | task result; not a timing proof |
| SmolVLA | `tools/smolvla_preflight.py`, `verify_smolvla_action_expert.py`, `verify_smolvla_cached_expert.py`, `verify_smolvla_cached_trajectory.py`, `flowedge-inspect` | Real checkpoint schema, suffix parity, and captured VLM-cache expert/Euler replay | preflight + PyTorch max-error + real-capture replay artifacts; no native VLM, timing, or control-quality claim |

## Canonical policy report

```text
python tools/benchmark/run_policy_report.py models/policy.safetensors \
  --source models/diffusion_pusht \
  --revision <immutable-hf-revision> \
  --steps 10 --iterations 100 --warmup 5 --threads 1 \
  --output bench/artifacts/policy/diffusion-pusht-report.json
```

The command writes JSON and Markdown together. The reference backend is the
actual LeRobot policy executed with PyTorch; the candidate is FlowEdge native.
The manual workflow `.github/workflows/policy-evaluation.yml` exposes the same
parameters and can target `ubuntu-latest` or a `self-hosted` hardware runner.

## Interpretation rules

| Claim | Allowed when | Not allowed when |
|---|---|---|
| PyTorch speedup | Same checkpoint, processor, host, threads, and schedule | Only a synthetic head or different preprocessing was timed |
| Real-time suitability | Full control period, delivery path, queue state, and deadline misses are measured | Only p50 or a one-shot component is available |
| Zero hot allocations | Native path allocation instrumentation is present | Python/PyTorch process counts are unknown |
| Hardware result | Artifact names board, firmware, driver, compiler, and parameters | A CPU result is presented as Tenstorrent/CUDA evidence |

Current backend status: **CPU available · Tenstorrent planned · CUDA planned**.
