# Checkpoint preflight

```{image} ../_static/figures/convert.svg
:alt: convert once
:class: fe-fig
```

`flowedge-inspect` validates a `.safetensors` checkpoint before it reaches an
inference service or robot.

```bash
flowedge-inspect models/mamba_flow.safetensors
flowedge-inspect models/mamba_flow.safetensors --json
```

The JSON output is versioned with `schema_version: 1` and contains the model
family, precision, tensor count, FlowEdge digest, backbone/action dimensions,
weight bytes, arena bytes, persistent decode-state bytes, Diffusion Policy
configuration fields, missing required tensors, unsupported tensors, and a
compatibility result. The human-readable
format is intended for deployment logs; JSON is intended for admission checks.

The command returns zero only for a supported checkpoint. It returns non-zero
for malformed safetensors, an unknown model family, missing required tensors,
incompatible dimensions, unsupported tensor dtypes, or a checkpoint that the
runtime weight loader cannot open.

The report estimates the existing fixed runtime arena. It does not start worker
threads or initialize persistent decode state for a deployed engine. Observation
preprocessing and model training remain outside FlowEdge.
