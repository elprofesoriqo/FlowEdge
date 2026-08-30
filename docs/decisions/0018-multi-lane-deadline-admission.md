# ADR 0018: Simulate actual worker lanes during deadline admission

## Context

Calibrated admission originally summed active and queued NFE as if Relay had one serialized worker.
That remained safe after the parallel pool landed, but rejected feasible requests whenever two or
more workers could meet the same deadline independently.

## Decision

`AdmissionContext` carries up to eight active lanes, each with its generation and remaining NFE, plus
the configured worker count. `HeadWorkerPool` snapshots this state atomically for every submission.

The bounded scheduler maintains a preallocated pointer workspace. For a candidate it:

1. initializes each worker's available time from non-cancellable active work;
2. combines the candidate with retained finite-deadline queued jobs;
3. orders those jobs by the existing EDF/freshness tie-break;
4. assigns each job to the earliest available worker; and
5. rejects unless every completion plus the fixed reserve fits its deadline.

Work from an older generation is excluded because accepting the candidate cancels it. Arithmetic
uses subtraction bounds before addition, so even maximum timestamps and costs cannot wrap into a
feasible result. The workspace reserves capacity at scheduler construction and does not allocate on
submission.

## Consequences

- Two jobs that cannot serialize before a deadline may be accepted when two idle workers can finish
  them independently.
- Busy lanes constrain only their own availability instead of consuming every worker's budget.
- The simulation is a conservative non-preemptive EDF list schedule; it does not claim an optimal
  multiprocessor schedulability proof.
- Backpressure and operating-system interference still require a deployment-specific reserve.
