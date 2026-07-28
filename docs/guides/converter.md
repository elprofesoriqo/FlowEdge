# Converter

FlowEdge loads a fixed tensor layout. `backbone.*` for the backbone. `flow.*` for the head. The converter maps a torch or HuggingFace checkpoint into that layout.

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

`REQUIRED` lists the tensors the model constructor looks for. The converter checks they are present and stops if not. Keep it in sync with the constructor.

If the source normalizes actions with dataset stats, carry those stats through and un-normalize the sampled action. Otherwise the output stays in normalized space.

Source: `convert/convert.py`. See [ADR 0006](../decisions/0006-obs-encoder-out-of-scope).
