# Deployment profile

FlowEdge checkpoints may carry a `flowedge.deployment_profile` value in the
`.safetensors` `__metadata__` map. The value is a compact JSON object describing
the model-to-runtime contract. Profile parsing is setup-time work; the runtime
does not allocate while sampling.

Version 1 uses this shape:

```json
{
  "profile_version": 1,
  "model_compatibility_version": 1,
  "observation_schema_hash": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "action_dim": 8,
  "action_horizon": 16,
  "action_units": "normalized",
  "normalization_type": "minmax",
  "normalization_parameters": {
    "min": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    "max": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  },
  "solver_default": "euler",
  "solver_min_steps": 6,
  "solver_max_steps": 16
}
```

The profile version and model compatibility version are independent. FlowEdge
currently accepts only version `1` for each. `action_dim` is in the range
`1..65536`, `action_horizon` is `1..4096`, and solver limits are in
`1..4096` with the minimum no greater than the maximum.

`observation_schema_hash` is an exact `sha256:` prefix followed by 64 hexadecimal
digits. `action_units` is `normalized` or `physical`; `normalization_type` is
`none` or `minmax`. A `minmax` profile must provide finite, equal-length `min`
and `max` arrays with one strictly increasing pair per action dimension. A
`none` profile must provide empty normalization parameters.

Profiles are descriptive metadata. FlowEdge validates the contract but does
not execute observation preprocessing. Checkpoints without a profile remain
valid through the legacy compatibility path until a future profile version
makes metadata mandatory.

The C++ contract is implemented in
`src/core/protocol/deployment_profile.h`. Diffusion conversion embeds this
profile from the LeRobot config and action statistics; runtime exposure and
preflight inspection use the same versioned metadata.
