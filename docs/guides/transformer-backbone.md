# Fixed-shape Transformer baseline

FlowEdge has an experimental general causal Transformer decoder beside Mamba.
It is a fixed-shape, batch-one CPU baseline for the GPT-style contract in
[issue #10](https://github.com/elprofesoriqo/FlowEdge/issues/10). It is not a
generic Hugging Face loader, a production backend, or a SmolVLA implementation.

## Contract

The checkpoint carries an explicit `transformer.config` F32 vector:

`[vocab, d_model, n_layers, n_heads, max_sequence, mlp_dim]`.

Tensor names are deliberately FlowEdge-native: token and learned-position
embeddings, pre-attention and pre-MLP affine LayerNorms, fused QKV projections,
attention output projections, GELU MLP projections, and final LayerNorm. All
KV buffers are allocated at initialization as `[layer][position][hidden]`.

`run(tokens)` resets and fills that cache. `step(token)` appends one key/value
per layer and attends only to positions through the current token. No synthetic
C++ checkpoint fixture is used as release evidence; an actual converted policy
and a matched reference replay are required before this baseline can be called
deployed.

The converter is exercised with the downloaded `sshleifer/tiny-gpt2` checkpoint
at revision `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be`. It compares a Hugging
Face hidden value against the native executable on prefix `[1,2,3,4]`:

```bash
python tools/verification/verify_transformer_reference.py \
  models/tiny-gpt2 models/tiny-gpt2.flowedge.safetensors \
  --binary build/transformer_forward \
  --output bench/artifacts/tiny-gpt2-transformer-reference.json
```

This validates a real checkpoint conversion and fixed-prefix execution only;
it is not an action-policy, latency, or SmolVLA result.

## SmolVLA preflight

SmolVLA is architecturally different from the baseline: it has a visual-language
encoder and an action expert with RMSNorm, RoPE, grouped-query self/cross
attention, and SwiGLU. `tools/smolvla_preflight.py` validates the exact tensor
layout of the pinned `lerobot/smolvla_base` checkpoint before conversion:

```bash
python tools/smolvla_preflight.py models/smolvla_base/model.safetensors \
  --config models/smolvla_base/config.json --hash \
  --output bench/artifacts/smolvla-base-preflight.json
```

The resulting manifest establishes the real-source contract only. It records
`supported_by_flowedge: false`; conversion, source-encoder parity, a complete
Euler action-expert path, and matched task evaluation remain required.

With the upstream VLM weights available locally, capture a separate source
policy-construction record:

```bash
python tools/verification/verify_smolvla_source.py models/smolvla_base \
  --output bench/artifacts/smolvla-base-upstream-load.json
```

This proves only that the pinned upstream `SmolVLAPolicy` can construct from
the local checkpoint. It is not FlowEdge inference or an action-quality result.
