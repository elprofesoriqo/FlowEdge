# FlowEdge LeRobot edge-inference RFC draft

## Proposal

Provide FlowEdge as an optional deployment adapter for fixed-shape LeRobot
policies. It consumes the already-preprocessed condition, preserves LeRobot
action chunk selection, and returns physical actions after MIN_MAX restoration.
Training, cameras, robot drivers, limits, and emergency stop remain LeRobot or
application-owned.

## Initial contract

| Area | Initial support | Boundary |
|---|---|---|
| Policy | pinned `diffusion_pusht` ConditionalUnet1D | no ACT/VLA claim |
| Artifact | `model.safetensors`, `config.json`, processor state | validate before conversion |
| Input | flattened preprocessed condition | encoder stays outside FlowEdge |
| Output | observation-prefix action chunk | robot limits stay outside FlowEdge |
| Runtime | native fixed-memory FlowEdge | no ONNX/ExecuTorch fallback yet |

## Evidence requested from design partners

1. A fixed policy directory and deterministic condition/action fixture.
2. PyTorch-versus-FlowEdge output tolerance on that fixture.
3. SO-100/SO-101 or simulator rollout with p50/p99, missed deadlines, RSS,
   startup time, and zero hot-path allocation report.
4. ARM64 target details: OS, compiler, CPU, RAM, and model revision.

## Open questions for LeRobot maintainers

- Is an out-of-tree `flowedge-lerobot` package the preferred integration form?
- Which processor contract should be treated as stable for deployment adapters?
- Which simulator/robot fixture is suitable for a first reproducible pilot?

## Non-goals

This proposal does not add training, ROS drivers, dynamic-shape execution,
Tenstorrent support, or universal policy compatibility. Unsupported artifacts
must fail before deployment with an actionable diagnostic.
