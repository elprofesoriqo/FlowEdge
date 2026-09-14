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
Face hidden states against native full-prefix and streaming execution on prefix
`[1,2,3,4]`:

```bash
python tools/verification/verify_transformer_reference.py \
  models/tiny-gpt2 models/tiny-gpt2.flowedge.safetensors \
  --binary build/transformer_forward \
  --output bench/artifacts/tiny-gpt2-transformer-reference.json
```

This validates real checkpoint conversion, full-prefix execution, and streaming
KV-cache parity only; it is not an action-policy, latency, or SmolVLA result.

Check that the real checkpoint's streaming path performs no heap allocation
after initialization:

```bash
build/flowedge_model_lifecycle_check models/tiny-gpt2.flowedge.safetensors \
  --stream 1 2 3 4 --cycles 3
```

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

## SmolVLA source-parity capture contract

Before implementing the action expert, export a source action chunk from a
real observation captured through the upstream LeRobot processor. The `.npz`
capture must contain batch-one, pre-processor tensors:

- `observation.state`: F32 `[1, 6]`;
- `observation.images.camera1`, `camera2`, and `camera3`: F32 `[1, 3, 256, 256]`
  RGB values in `[0, 1]`;
- `observation.language.tokens` and `observation.language.attention_mask`: I64
  `[1, 48]`;
- `noise`: F32 `[1, 50, 32]`, captured once and reused by every implementation.

```bash
python tools/verification/export_smolvla_reference.py \
  models/smolvla_base real-observation.npz \
  --output smolvla-source-action.npz \
  --manifest smolvla-source-action.json
```

The exporter does not synthesize defaults and rejects missing, reshaped, or
non-RGB-range inputs. Its action chunk and JSON digests form the later
FlowEdge parity target.

The native inspector recognizes this real checkpoint schema but fails closed:

```bash
build/flowedge-inspect models/smolvla_base/model.safetensors --json
```

The report identifies `family: "smolvla"` and explains that the LeRobot
preprocessing boundary, VLM encoder, and action expert are not implemented.
This is a schema/provenance check, not a claim of native SmolVLA support.

For a portable evidence artifact, run the verifier against the same binary:

```bash
python tools/verification/verify_smolvla_inspection.py \
  models/smolvla_base/model.safetensors \
  --binary build/flowedge-inspect \
  --output bench/artifacts/smolvla-base-inspection.json
```
