# Action delivery

Use `ActionDeliveryGate` between Relay output and the controller.

```{mermaid}
flowchart LR
  R[ActionMessage] --> C[Chunk view]
  C --> F[Model + session + freshness]
  F --> O[Timed overlap]
  O --> S[Bounds + rate limit]
  S --> A[Controller action]
```

| Input | Gate behavior |
|---|---|
| New generation | Accept or replace the active chunk |
| Old generation/sequence | Reject as stale |
| Different session/model | Reject |
| Late chunk or old observation | Reject as expired |
| Unsafe value | Reject the chunk or clamp each control step |
| Missed control tick | Skip obsolete steps by timestamp |

```cpp
auto gate = fe::relay::ActionDeliveryGate::create(
    model,
    {.control_dim = 7,
     .overlap_steps = 2,
     .step_period_ns = 10'000'000,
     .max_source_age_ns = 80'000'000,
     .unsafe_policy = fe::relay::UnsafeActionPolicy::kClamp},
    {.lower = lower, .upper = upper, .max_delta_per_step = max_delta});

gate->accept(action_message, now_ns);
if (auto step = gate->next(now_ns))
  controller.write(step->values);
```

`action_dim` is the flattened chunk size: `step_count × control_dim`. Setup copies limits once;
`accept`, replacement, overlap, and `next` allocate nothing. Run `action_delivery_sample` for a
25 Hz policy feeding a 100 Hz controller.

This is a deterministic software gate, not a certified robot safety controller. Hardware limits,
watchdogs, and emergency stops remain downstream.
