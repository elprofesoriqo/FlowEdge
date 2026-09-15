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

The C and Python APIs also expose `run_embeddings`, which accepts caller-owned
`[sequence, d_model]` F32 rows and runs the same reset-and-prefill path. An
optional batch-one prefix `attention_mask` (`1` values followed by padding `0`
values) makes the condition-sequence boundary explicit for VLA adapters. This
is groundwork for SmolVLA; it does not implement cross-attention, the VLM, or
the current SmolVLA action expert.

The converter is exercised with the downloaded `sshleifer/tiny-gpt2` checkpoint
at revision `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be`. It compares a Hugging
Face hidden states against native full-prefix and streaming execution on prefix
`[1,2,3,4]`:

```bash
python tools/verification/verify_transformer_reference.py \
  models/tiny-gpt2 models/tiny-gpt2.flowedge.safetensors \
  --binary build/transformer_forward \
  --output bench/artifacts/transformer/tiny-gpt2-transformer-reference.json
```

This validates real checkpoint conversion, full-prefix execution, and streaming
KV-cache parity only; it is not an action-policy, latency, or SmolVLA result.

Verify the external-embedding boundary against the same real checkpoint (the
Python module must be built with `FLOWEDGE_PYTHON=ON`):

```bash
python tools/verification/verify_transformer_embeddings.py \
  models/tiny-gpt2.flowedge.safetensors --module-path build \
  --output bench/artifacts/transformer/tiny-gpt2-transformer-embeddings.json
```

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
  --output bench/artifacts/smolvla/smolvla-base-preflight.json
```

The resulting manifest establishes the real-source contract only. It records
`supported_by_flowedge: false`; the native path is intentionally partial until
the source encoder, captured-VLM expert replay parity, complete Euler action
integration, and matched task evaluation are available.

The first native action-expert boundary is now implemented for the real
checkpoint: action projection, the exact SmolVLA sine/cosine timestep embedding,
the time MLP, and checkpoint-derived dimensions. Verify it against PyTorch with:

```bash
python tools/verification/verify_smolvla_action_expert.py \
  models/smolvla_base/model.safetensors \
  --module-path build-research \
  --output bench/artifacts/smolvla/smolvla-action-expert-suffix.json
```

This artifact is projection parity only. It does not execute the VLM image or
language encoder, cached-VLM self/cross-attention replay, or a policy rollout.

With the upstream VLM weights available locally, capture a separate source
policy-construction record:

```bash
python tools/verification/verify_smolvla_source.py models/smolvla_base \
  --output bench/artifacts/smolvla/smolvla-base-upstream-load.json
```

This proves only that the pinned upstream `SmolVLAPolicy` can construct from
the local checkpoint. It is not FlowEdge inference or an action-quality result.

## SmolVLA source-parity capture contract

To validate the captured VLM cache and native action expert, export a source action chunk from a
real observation captured through the upstream LeRobot processor. The `.npz`
capture must contain batch-one, pre-processor tensors:

- `observation.state`: F32 `[1, 6]`;
- `observation.images.camera1`, `camera2`, and `camera3`: F32 `[1, 3, 256, 256]`
  RGB values in `[0, 1]`;
- `observation.language.tokens`: I64 `[1, 48]`; `observation.language.attention_mask`:
  boolean `[1, 48]`, emitted by LeRobot's tokenizer.
- `noise`: F32 `[1, 50, 32]`, captured once and reused by every implementation.

For cached-expert replay, add `noisy_actions` (F32 `[1, 50, 32]`) and
`timestep` (F32 `[1]`); they identify one actual source denoising step.

```bash
python tools/verification/export_smolvla_reference.py \
  models/smolvla_base real-observation.npz \
  --output smolvla-source-action.npz \
  --manifest smolvla-source-action.json
```

The exporter does not synthesize defaults and rejects missing, reshaped, or
non-RGB-range inputs. Its action chunk and JSON digests form the later
FlowEdge parity target.

### Cached-VLM action-expert replay

For the native expert boundary, preserve one real denoising input in the same
capture as `noisy_actions` (`F32 [1, 50, 32]`) and `timestep` (`F32 [1]`). The
exporter builds the upstream prefix once, retains its RoPE-applied VLM K/V
cache, and records the expected expert hidden state and velocity. It does not
fill in missing values:

The source cache is `[1, prefix, 5, 64]` per layer and is transported to
FlowEdge as `[16, prefix, 320]`. Its validity mask may be sparse: LeRobot
right-pads language tokens before appending the valid state token. The replay
verifier therefore compares BF16 hidden states with a `0.1` maximum-error
default and keeps the action-velocity threshold at `0.02`.

```bash
python tools/verification/export_smolvla_expert_reference.py \
  models/smolvla_base real-observation.npz \
  --output smolvla-expert-reference.npz \
  --manifest smolvla-expert-reference.json

python tools/verification/verify_smolvla_cached_expert.py \
  models/smolvla_base/model.safetensors smolvla-expert-reference.npz \
  --module-path build-research \
  --output bench/artifacts/smolvla/smolvla-cached-expert.json
```

The report is the required evidence for the cached-VLM action-expert path. It
does not validate preprocessing, VLM encoding, the full ten-step flow solve,
timing, or control quality.

### Cached-VLM Euler trajectory replay

The native `smolvla_sample` call implements the source's deterministic Euler
schedule (`t = 1 - step / N`, `x += -v / N`) from caller-owned noise. Add the
same real `noise` input used by the upstream action call to produce a complete
trajectory target from the cached VLM prefix:

```bash
python tools/verification/export_smolvla_expert_reference.py \
  models/smolvla_base real-observation.npz \
  --output smolvla-expert-reference.npz \
  --trajectory-output smolvla-trajectory-reference.npz

python tools/verification/verify_smolvla_cached_trajectory.py \
  models/smolvla_base/model.safetensors smolvla-trajectory-reference.npz \
  --module-path build-research \
  --output bench/artifacts/smolvla/smolvla-cached-trajectory.json
```

This establishes only expert-side Euler replay from an externally generated
cache. It remains short of native full-policy inference until the observation
processor and VLM prefix/cache producer have matched source parity.

The native inspector recognizes this real checkpoint schema but fails closed:

```bash
build/flowedge-inspect models/smolvla_base/model.safetensors --json
```

The report identifies `family: "smolvla"` and explains that the LeRobot
preprocessing boundary and VLM encoder are not implemented. The action expert
can consume an external cache, but this remains a schema/provenance check—not
a claim of native full SmolVLA support.

For a portable evidence artifact, run the verifier against the same binary:

```bash
python tools/verification/verify_smolvla_inspection.py \
  models/smolvla_base/model.safetensors \
  --binary build/flowedge-inspect \
  --output bench/artifacts/smolvla/smolvla-base-inspection.json
```
