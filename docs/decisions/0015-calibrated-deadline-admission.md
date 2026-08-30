# ADR 0015: Make deadline admission calibrated and observable

## Status

Accepted.

## Context

Dropping requests only after their deadlines expire wastes compute and leaves a producer unable to
distinguish overload from transport failure. A fixed latency guess is also unsafe: function-evaluation
cost changes with the checkpoint, precision, host, thread count, affinity, and power mode. Relay
already carries remaining NFE and uses EDF, so it can make a bounded schedulability decision when the
deployment supplies a measured cost.

## Decision

Keep admission disabled by default. Enable it with `flowedge-relayd --nfe-ns N`, where `N` is a
deployment-measured upper bound for one model function evaluation. An optional
`--admission-reserve-ns N` adds a fixed safety reserve.

For a finite-deadline request, test every affected EDF deadline prefix. The demand includes the new
request, retained queued requests whose deadlines fall in that prefix, and active work that will not
be cancelled by the new generation. Use saturating arithmetic. Reject the new request without
advancing the freshness generation or pruning feasible queued work when any affected prefix cannot
fit before its deadline. A newer accepted generation excludes older work because that work will be
cancelled before another complete solver step.

Return valid, model-compatible `ActionMessage` outcomes for observable scheduling failures. The
envelope action code distinguishes `rejected_stale`, `rejected_deadline`, `rejected_capacity`, and
`expired` from an inference action. Rejection metadata retains the request sequence, session,
generation, model digest, dimensions, deadline, and remaining NFE. Invalid or model-incompatible
input is still discarded because Relay cannot safely reflect untrusted identifiers.

## Consequences

- Producers can react differently to overload, stale generations, expiry, and inference failure.
- Admission behavior remains unchanged unless a deployment explicitly supplies calibration.
- Full-queue replacement reports the displaced request as `rejected_capacity`; a request that cannot
  displace earlier work receives the same code itself.
- Rejection delivery is bounded by the action SPSC ring. If its consumer stops draining, the daemon
  increments `rejection_outputs_dropped` rather than allocating an unbounded side queue.
- `flowedge_relay_bench` reports a conservative end-to-end p99 nanoseconds-per-NFE starting point.
  Production calibration still needs representative traces, thermal state, affinity, and safety margin.
