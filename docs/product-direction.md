# Product Direction

FlowEdge owns the deployment boundary between a trained action policy and robot control code. It
should make that boundary predictable, measurable, and portable without becoming a training or graph
runtime.

## Product rule

| Keep | Leave outside FlowEdge |
|---|---|
| Fixed model implementations and shared kernels | Training loops and dataset readers |
| Checkpoint, memory, timing, and replay contracts | Vision and observation encoders |
| Local bounded scheduling and transport | Distributed orchestration |
| C/C++/Python and thin ecosystem adapters | Robot-specific safety controllers |

## Deployment contracts available now

| Surface | User outcome | Start here |
|---|---|---|
| Checkpoint preflight | Reject incompatible artifacts before a robot starts | [Preflight](guides/checkpoint-preflight) |
| Deployment profile | Bind dimensions, units, normalization, and solver limits to weights | [Profile contract](architecture/deployment-profile) |
| Deadline profile | Gate tail latency and hot-path allocations for one host | [Deadline guide](guides/deadline-profile) |
| Trace and state capsule | Reproduce actions and migrate compatible state | [Relay proposal](ecosystem/relay-proposal) |
| LeRobot adapter | Run a converted Diffusion Policy through a bounded rollout seam | [LeRobot](guides/lerobot) |

## Next customer sequence

```{mermaid}
flowchart LR
  P[Publish measured edge artifact] --> H[Run one hardware pilot]
  H --> U[Propose LeRobot integration]
  U --> M[Expand model import matrix]
  M --> B[Add measured hardware backends]
```

| Order | Deliverable | Proof |
|---:|---|---|
| 1 | Reproducible FlowEdge-versus-LeRobot artifact | Same checkpoint, processor, host, and commands |
| 2 | SO-100/SO-101 deployment pilot | Deadline, RSS, action shape, and stop behavior captured |
| 3 | Upstream LeRobot RFC | Narrow adapter boundary accepted by maintainers |
| 4 | Portable import/backend matrix | Each supported path has conversion and runtime verification |
| 5 | Accelerator work | Same contract and benchmark pass on the target hardware |

Success means an external team can convert a policy, verify it in CI, and reproduce a robot-side
failure without adopting FlowEdge-specific training code.
