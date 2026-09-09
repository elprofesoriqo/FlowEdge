# Converter

FlowEdge loads fixed tensor layouts: `backbone.*` for the backbone, `flow.*` for
the flow head, and `dp.*` for the Diffusion Policy head. The converter maps a
PyTorch or Hugging Face checkpoint into those layouts.

```{mermaid}
%%{init: {'theme':'base','flowchart':{'htmlLabels':false,'nodeSpacing':28,'rankSpacing':34,'useMaxWidth':false},'themeVariables':{'primaryColor':'#f6ead0','primaryBorderColor':'#7b2733','lineColor':'#7b2733','primaryTextColor':'#2b2521','secondaryColor':'#eaddbf','tertiaryColor':'#faf3e2','fontFamily':'system-ui, -apple-system, Segoe UI, Roboto, sans-serif','fontSize':'13px'}}}%%
graph LR
  SRC["torch / HF checkpoint"] --> MAP["arch mapping"]
  MAP --> VAL["validate required tensors"]
  VAL --> ST[".safetensors"]
  ST --> ENG[Engine]
```

```bash
python convert/convert.py <source> models/out.safetensors --arch mamba
```

Extract only the action head for an external encoder:

```bash
python convert/convert.py models/full.safetensors models/head.safetensors \
  --arch mamba --component head --dtype bf16
```

`--component backbone` drops the head; `all` preserves both. Head-only output validates the four
required flow projections and is directly loadable through `fe_engine_sample_condition`.

## A mapping

An architecture is one function in the `ARCH` registry. It takes the source state dict and returns a dict keyed by FlowEdge names.

```python
def mamba(sd):
    out = {}
    for k, v in sd.items():
        k = k.replace("backbone.embedding.weight", "backbone.embeddings.weight")
        if k.startswith("backbone.") or k.startswith("flow."):
            out[k] = v
    return out

ARCH = {"mamba": mamba}
```

FlowEdge uses PyTorch `[out, in]` linear layout, so most weights need no transpose. Rename keys, drop what the engine does not use, and transpose only where the source layout differs.

## Validation and normalization

`REQUIRED_BACKBONE` and `REQUIRED_FLOW` list the tensors the model constructors look for. The
converter checks the selected component and stops if it is incomplete. Keep these lists in sync
with the constructors.

With `--dtype bf16`, only two-dimensional matmul weights are narrowed. Embeddings, normalization,
biases, convolution weights, `A_log`, and `D` remain F32 because the current kernels and model
constructors require them in that format.

If the source normalizes actions with dataset stats, carry those stats through and un-normalize the sampled action. Otherwise the output stays in normalized space.

## LeRobot Diffusion Policy

Download `lerobot/diffusion_pusht`, then pass its directory so the converter can
read both `model.safetensors` and `config.json`:

```bash
hf download lerobot/diffusion_pusht --revision 84a7c23178445c6bbf7e1a884ff497017910f653 \
  --local-dir models/diffusion_pusht
python convert/convert.py models/diffusion_pusht \
  models/diffusion_pusht.flowedge.safetensors --arch diffusion
```

This maps `ConditionalUnet1D`, shortens tensor names for the fixed loader,
stores horizon/dimension/scheduler metadata, and preserves action min/max. It
drops the RGB encoder by design. Unsupported schedules, prediction modes,
normalization, or U-Net variants fail with a specific conversion error.

Source: `convert/convert.py`. See [ADR 0006](../decisions/0006-obs-encoder-out-of-scope).
