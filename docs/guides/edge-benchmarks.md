# Reproducible edge benchmarks

Use one artifact when comparing the LeRobot/PyTorch reference with FlowEdge. It is designed for
an issue, release note, or partner hand-off: reviewers can reproduce the exact checkpoint and
processor contract without treating one machine's result as a promise for every robot.

```{mermaid}
flowchart LR
  C[Same checkpoint + revision] --> P[Same processor stats]
  P --> I[Same observation/action contract]
  I --> L[LeRobot/PyTorch run]
  I --> F[FlowEdge run]
  L --> A[JSON artifact]
  F --> A
  A --> R[Markdown report]
```

## Capture

```bash
cp bench/artifacts/lerobot-vs-flowedge.template.json bench/artifacts/run.json
# Fill run.json with the checkpoint hash, processor stats, host, commands, and both measurements.
python tools/benchmark/benchmark_artifact.py bench/artifacts/run.json \
  --report bench/artifacts/run.md
```

The template is intentionally `not_measured`; it cannot be published as a result until both
paths are captured. If LeRobot cannot run on the host, set its status to `unavailable` and record
the reason. The report will keep the missing comparison visible.

## Measurement contract

| Field | Meaning | Rule |
|---|---|---|
| `startup_ms` | Process/checkpoint startup to ready | State whether the process was cold or warm in `limitations` |
| `encoder_ms` | Observation encoding and processor work | Report p50, p95, p99 separately |
| `policy_ms` | Policy inference/solver only | Report p50, p95, p99 separately |
| `end_to_end_ms` | Encoder + policy + adapter boundary | Use for control-loop budgeting |
| `throughput_hz` | Completed end-to-end iterations per second | Include thread count and batch size in `contract`/`limitations` |
| `rss_mb` | Peak or sampled process resident set size | Name the measurement tool and timing in `limitations` |
| `allocations` | Setup and hot-path allocation counts | FlowEdge hot path must remain zero after setup |

The artifact also requires model and processor revisions, a checkpoint SHA-256, canonical
observation schema hash, normalization/action units, hardware/compiler/build details, and the
exact commands. `tools/benchmark/benchmark_artifact.py` rejects missing fields or non-monotonic quantiles.

## Interpreting a report

| Safe conclusion | Unsafe conclusion |
|---|---|
| "On this host and revision, FlowEdge policy p99 was ..." | "FlowEdge is always faster" |
| "The two paths used the same processor stats." | "The result applies to every SO-100/SO-101 setup." |
| "Hot-path allocations were zero after setup." | "Checkpoint loading allocates zero memory." |

Publish the artifact and report together. Keep raw benchmark output and environment details next
to them when a partner needs to reproduce the run.
