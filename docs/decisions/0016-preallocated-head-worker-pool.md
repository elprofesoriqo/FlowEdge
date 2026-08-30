# ADR 0016: Run independent head engines in a preallocated worker pool

## Status

Accepted.

## Context

The first daemon advanced one Core engine from its transport event loop. Cooperative steps bounded
cancellation latency, but independent requests could not execute concurrently and model compute could
delay ring draining. Merely interleaving two engines on the same event-loop thread would increase
in-flight capacity without increasing compute throughput.

## Decision

Add `HeadWorkerPool`, configured by `flowedge-relayd --workers 1..8` and defaulting to one worker.
Each slot owns:

- one fully loaded `HeadWorker`/Core engine;
- fixed-capacity condition and action storage;
- one dedicated outer C++ worker thread;
- atomic request, generation, remaining-NFE, cancellation, and lifecycle state.

The daemon event loop remains the only producer/consumer for shared-memory rings. It copies an EDF
dispatch into an idle slot and immediately returns to transport work. A worker runs the complete
cooperative solve on its own thread, checking Core generation cancellation between solver steps, then
publishes a ready state with release/acquire ordering. The slot retains its action while the output
ring is full, so backpressure does not overwrite results or block other workers.

Keep Core background threads independently configurable per engine through `--threads`. A typical
CPU throughput configuration begins with `--workers 2 --threads 0`; using several outer workers and
several Core thread pools simultaneously must be justified by measurement to avoid oversubscription.

## Consequences

- Independent requests execute concurrently while one event loop preserves SPSC ring ownership.
- Requests, results, worker slots, and engines allocate only during startup; the pool benchmark checks
  for zero allocations after warm setup.
- A newer generation atomically cancels running older work and drops older results that were already
  waiting under output backpressure.
- Deadline admission conservatively counts aggregate active NFE as single-lane work. This cannot
  overpromise a deadline, although it may under-admit when several workers are available.
- This first implementation loaded a complete model and arena per worker. ADR 0017 supersedes that
  ownership detail with one shared immutable checkpoint store plus private mutable worker arenas.
- Worker failures are counted and reported without terminating healthy slots or growing an error queue.
