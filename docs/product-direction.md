# Product direction

## Position

```{mermaid}
flowchart LR
  Train[Training stack] -->|checkpoint + metadata| Preflight[FlowEdge preflight]
  Preflight --> Runtime[Predictable Core runtime]
  Runtime --> Relay[Optional local Relay]
  Relay --> Robot[Robot / simulator]
  Runtime --> Replay[Replay + timing evidence]
```

| FlowEdge owns | Integrator owns |
|---|---|
| Checkpoint validation and fixed inference | Training and dataset pipelines |
| Static runtime state and solver execution | Camera/state encoders |
| State identity, migration, traces, metrics | Robot safety controller |
| Deadline admission and action gate | OS/board real-time configuration |

## Highest-leverage additions

| Feature | Why it matters | Smallest useful form |
|---|---|---|
| Checkpoint preflight | Fail before a model reaches a robot | Schema, precision, dimensions, arena, unsupported tensors |
| Replay capsule | Reproduce a surprising action | Model digest, encoded input, noise, solver, output, timing |
| Deadline profile | Turn latency into a deployment contract | p50/p99/p999, max, allocations, CPU affinity |
| Deployment profile | Share training/runtime assumptions | Units, normalization, observation hash, horizon, solver |

## Partner path

```{mermaid}
flowchart TD
  A[Stable C/Python contract] --> B[LeRobot adapter]
  B --> C[One supported robot rollout]
  C --> D[Trace + latency + allocation evidence]
  D --> E[Partner feedback]
  E --> F[Only then: backend or middleware expansion]
```

The product stays narrow: validate the deployed artifact, execute the fixed policy predictably, and
make failures reproducible.
