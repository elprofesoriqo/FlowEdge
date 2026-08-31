# Product Direction

FlowEdge's advantage is predictable action generation at the point where a trained policy meets
real hardware. It should not compete with training frameworks, dataset tools, or a broad graph
runtime. The highest-leverage additions are small, verifiable deployment contracts around the
existing C ABI.

The initial Relay implementation now supplies model identity, portable action traces, calibrated
deadline admission, generation cancellation, and bounded parallel workers. The items below describe
the remaining product surface rather than unimplemented research ideas.

## Recommended Differentiators

### Checkpoint preflight report

Provide a host-side command that reads a checkpoint without constructing a live engine and emits:
model family, tensor schema, action dimensions, precision, static arena bytes, estimated persistent
state, and unsupported tensors. A startup can put this report in CI before a model reaches a robot.
It makes the current self-describing checkpoint format operationally useful and prevents a failed
deploy from becoming a field debugging session.

### Replay capsule

Define a small, versioned artifact containing the model digest, tokens or already-encoded condition,
noise, solver, action output, build revision, and timing summary. The engine stays deterministic, so
the capsule becomes a portable reproducer for a surprising action across a robot, simulator, and CI.
This is especially valuable to early-stage teams that cannot afford an extensive observability stack.

Relay trace v2 implements the deterministic condition/action core. Generic cooperative jobs now add
checksummed, model- and schema-bound state capsules for exact worker migration. Build revision and
step-level scheduling events remain to be added to complete a single incident artifact.

### Deadline budget contract

Expose a host-side benchmark profile with p50, p99, p999, maximum, allocation count, and CPU
affinity for the exact checkpoint and solver. Let deployment define a control period and fail the
profile when tail latency exceeds the budget. The novelty is treating a policy's deadline as an API
contract, rather than publishing only throughput numbers.

Relay already accepts a deployment-calibrated nanoseconds-per-NFE bound. Persistent profiles with
p999, affinity, thermal context, and automatic safety-margin selection remain future work.

### Deployment profile

Package checkpoint metadata beside the weights: action units, normalization parameters, observation
schema hash, action horizon, solver default, allowed solver range, and model compatibility version.
The runtime need only validate and expose this data; training still owns it. That creates a clean
handshake between a startup's training code and its robot integration without making FlowEdge an
observation encoder.

## Deliberately Out Of Scope

These ideas should not pull vision encoders, dataset readers, training loops, or arbitrary graph
execution into FlowEdge. Those are separate systems with different latency and dependency budgets.
The engine should stay narrow: validate the deployed artifact, run its fixed policy predictably, and
make that execution easy to reproduce and measure.
