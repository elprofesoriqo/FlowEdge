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
| LeRobot adapter | `--policy.type=flowedge` or `flowedge_smolvla` with hold/drop/raise | [LeRobot](guides/lerobot) |
| Trace and state capsule | Reproduce actions and migrate compatible state | [Relay proposal](ecosystem/relay-proposal) |

## Active development sequence

The [roadmap](roadmap) tracks research tracks. The product sequence is: make
Diffusion Policy faster than PyTorch CPU, finish the SmolVLA cached-expert
plugin, then measure one ARM/Jetson host. Relay, Tenstorrent, and DeadlineFlow
stay optional until a user policy wins.

```{mermaid}
flowchart LR
  P[Pack DP conv through matmul] --> S[SmolVLA policy type]
  S --> J[Jetson / SO-100 JSON]
  J --> U[Propose LeRobot integration]
```

| Order | Deliverable | Proof |
|---:|---|---|
| 1 | Packed Diffusion Policy kernels | Same-host replay at or below LeRobot/PyTorch, zero hot-path alloc |
| 2 | `flowedge_smolvla` plugin | Hybrid action chunk within existing envelopes |
| 3 | ARM/Jetson rollout JSON | Volunteer host, miss behavior, RSS |
| 4 | Wheels | `pip install flowedge` without a local C++23 toolchain |
| 5 | Accelerator work | Same contract on one measured device |

Success means an external team can convert a policy, verify it in CI, and reproduce a robot-side
failure without adopting FlowEdge-specific training code.
