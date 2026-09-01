# ADR 0010: External Conditions and Cooperative Solvers

## Status

Accepted

## Context

The original runtime coupled the flow head to the built-in token backbone and ran every ODE step in
one call. Real deployments often already own a vision or language encoder in another runtime, and a
robot scheduler needs bounded work units so fresh observations and safety tasks are not delayed by
an old action request.

## Decision

- Permit head-only checkpoints and expose the head's condition dimension.
- Accept caller-owned condition, noise, and output buffers through the C ABI.
- Keep the projected condition and all solver stages in fixed engine-owned workspace.
- Split integration only at complete solver-step boundaries.
- Preserve a monolithic convenience call implemented through the same resumable state machine.
- Keep scheduling, cancellation policy, transport, and model orchestration outside the engine.

## Consequences

- Existing PyTorch, TensorRT, ONNX Runtime, and custom encoders can use FlowEdge as a lightweight
  deterministic action decoder.
- A caller can budget inference in NFE-sized quanta without allocations or numerical drift.
- One engine handle is stateful and not concurrently callable.
- Transport and fleet-level scheduling need a separate component rather than dependencies in the
  core library.
