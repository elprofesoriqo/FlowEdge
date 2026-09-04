# ADR 0029: Bound job QoS and worker supervision

**Status:** Accepted

**Date:** 2026-09-03

## Context

A saturated FIFO can let background work consume every queue slot. A repeatedly failing adapter can
also keep receiving work and hide a bad lane behind aggregate failure counts.

## Decision

Keep EDF as the primary order. Add best-effort, interactive, and critical classes as equal-deadline
tie-breakers, with configured queue slots reserved for higher classes. Reject protected-capacity
violations with a typed `rejected_qos` result.

Count consecutive execution failures per lane. At the configured threshold, quarantine the lane only
when removal cannot strand accepted work. Quarantine uses the existing drain path; re-entry requires
an explicit recovery command after the lane reaches drained state.

Trace the service class on each job event. Expose QoS rejections, lane masks, failures, and
quarantines through the control protocol, Prometheus, JSON, and OTLP JSON.

## Consequences

- Critical capacity remains available during lower-class overload.
- Deadline feasibility still dominates service class.
- A failing lane cannot silently re-enter service.
- Reservations and quarantine state use fixed setup-time storage; admission remains allocation-free.
- Operators must choose reserves and thresholds for each deployment.
