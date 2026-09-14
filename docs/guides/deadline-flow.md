# DeadlineFlow execution plans

`protocol/deadline_flow.h` implements an allocation-free C++ selector with at most 32
plans and one selector per execution lane. Python exposes `ExecutionPlan`, `PlanContext`,
and `DeadlineFlow`. Each plan represents a complete, validated integration schedule with
declared latency, quality, memory, and hardware requirements.

Selection maximizes declared quality subject to deadline slack, buffered-action time,
memory, hardware compatibility, health, reserve, and switch cost. Completed calls update
latency estimates immediately for slowdowns; subsequent observations decay toward the
calibrated floor. This is a heuristic, not a probabilistic or hard deadline guarantee.
No feasible plan returns an empty selection for the application's explicit fallback.

## CPU evaluation bridge

`flowedge_lerobot.deadline.CpuPlanSelector` executes FP32 DDIM plans in the periodic
runner. Portfolio JSON contains a `plans` array; each entry requires `id`, `steps`,
`latency_ms`, and `quality`. Optional fields are `memory_bytes`, `solver`, `precision`,
and `trace_id`. Quality must be a measured score in [0, 1], not an assumed function of
step count. Latency includes dispatch-to-ready execution on the deployment host.

Use `flowedge-lerobot-evaluate ... --period-ms 100 --plans plans.json`. The first listed
plan bootstraps outside the periodic interval. Later choices use occupancy and observed
latency. Compare against fixed plans with identical seeds, periods, and fallbacks.

The generic contract carries precision, solver, and trace identifiers for executor
validation. The CPU bridge rejects non-FP32 precision, non-DDIM solvers, and accelerator
traces. TTNN/TT-Metalium execution, online quality estimation, live hardware telemetry,
and dynamic memory/sharding optimization are not implemented. Relay EDF admission
remains separate; the selector is not automatically enabled in its daemon.

Fewer complete steps traverse the full integration interval. Cancelling an unfinished
trajectory must not publish its noisy intermediate state as a completed cheaper plan.
