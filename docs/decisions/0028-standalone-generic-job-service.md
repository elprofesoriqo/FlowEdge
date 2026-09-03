# ADR 0028: Separate generic-job data and administration planes

**Status:** Accepted

**Date:** 2026-09-03

## Context

`JobService` already provided bounded process transport, but applications had to own its event loop.
Production operation also needs status, rolling lane drain, resume, and acknowledged shutdown even
when the job result ring is backpressured.

## Decision

Build `flowedge-jobd` and `flowedge-jobctl` on `FlowEdge::Relay`. Use one SPSC request/result pair for
jobs and a separate fixed-record SPSC pair for administration. Validate every control record, retain
responses under backpressure, and stop only after a shutdown acknowledgement is published.

The daemon provisions one Mamba adapter per lane over shared immutable weights. Its coordinator owns
both services, drains bounded lifecycle events into traces and metrics, and performs no hot-path
allocation. The public `JobControlClient` and `JobControlService` remain usable by custom embedders.

## Consequences

- Data backpressure cannot block administration.
- Rolling drain and status now cross a real process boundary on Windows and POSIX.
- The built-in daemon exposes Mamba streaming first; external runtimes remain optional adapters.
- Ring pairs remain SPSC. Multi-client admission belongs in an explicit gateway, not hidden locks.
