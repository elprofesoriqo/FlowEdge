# ADR 0017: Share immutable weights and place outer workers explicitly

## Context

The first parallel Relay pool constructed a complete Core engine from the checkpoint for every
worker. That kept ownership simple, but duplicated the dominant read-only tensor storage and made
memory grow approximately with the worker count. It also left outer worker threads to the operating
system even on multi-node hosts.

## Decision

Core exposes an opaque `fe_weights` handle. Loading it parses and copies supported checkpoint tensors
into one immutable, reference-counted store. `fe_engine_create_from_weights` constructs independent
engines that retain that store while allocating their own arena, thread pool, decode state, flow
workspace, cancellation generation, and load-time derived constants. Releasing the original handle
does not invalidate existing engines.

`HeadWorkerPool` loads one handle and uses it for every slot. Its `shared_weight_bytes()` result makes
the retained checkpoint cost observable. Hot inference still dereferences the same direct tensor
views and performs no reference-count operations or allocation.

Relay outer workers optionally bind with `--placement compact` or `--placement spread`; `none`
remains the portable default. Linux intersects placement with the process affinity mask and reads
CPU-to-node topology from sysfs. Windows uses processor-group-aware NUMA masks. Compact placement
fills nearby CPUs first; spread rotates workers across nodes before taking another CPU from a node.

## Consequences

- Worker count no longer multiplies checkpoint tensor storage.
- Each worker still owns mutable and transformed state, so the pool remains race-free and cancellation
  does not require locks around model execution.
- The public handle is useful to non-Relay embedders that create one engine per session.
- Shared weights can create remote-node reads when workers are spread. Deployments should compare
  compact and spread on representative checkpoints; no external NUMA library is required.
- Per-node replicated weight stores and OS-specific memory policy remain optional future tuning, not
  part of the default ownership contract.
