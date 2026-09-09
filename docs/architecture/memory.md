# Memory arena

## Allocation model

```{mermaid}
flowchart LR
  Slab[Fixed byte slab] --> W[Immutable weights]
  Slab --> P[Persistent decode / flow state]
  Slab --> R[Worker ring + objects]
  Slab --> S[Scratch]
  S -->|mark / reset| Reuse[Reuse next layer]
```

| Region | Lifetime | Owner |
|---|---|---|
| Weights | Engine/checkpoint lifetime | Shared immutable store |
| Decode state | Engine lifetime | One mutable stream |
| Flow state | Active solve | One engine |
| Worker objects/ring | Pool lifetime | Relay/runtime |
| Scratch | One layer/call | Reset to mark |

## Sizing

```text
slab = weight_bytes + decode_state + flow_state + worker_ring + alignment
```

The loader sizes the slab before construction. Exhaustion fails at load; the control loop cannot
silently grow memory. BF16 weights use two bytes per element and widen during compute.

## Hot-path contract

| Operation | Behavior |
|---|---|
| `alloc(n, alignment)` | O(1) aligned cursor bump |
| `mark()` / `reset_to(mark)` | Stack-like scratch reuse |
| Alignment | 64-byte cache/SIMD boundary |
| Load | May allocate loader metadata, weights, and slab |
| Inference and Relay hot paths | Zero heap allocations after setup |
| Multiple engines | Share immutable weights; keep state/scratch private |

Source: `src/core/arena/arena.h`.
