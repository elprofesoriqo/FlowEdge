# convert — checkpoint → FlowEdge `.safetensors`

FlowEdge loads a fixed tensor layout: `backbone.*` for the SSM and (optionally)
`flow.*` for the action head. This tool maps a torch / HuggingFace checkpoint into
that layout so you can run *your* model without hand-editing tensors.

```bash
pip install torch safetensors
python convert/convert.py <source> models/my_model.safetensors [--arch mamba] [--dtype f32|bf16]
```

`<source>` may be `.safetensors`, `.pt`, `.pth`, or `.bin` (a state dict; common
`state_dict`/`model`/`module` wrappers are unwrapped). The converter validates that
the output contains every tensor the engine needs and prints a summary:

```
wrote models/my_model.safetensors: 265 tensors, 24 layers, head=flow, dtype=f32
```

## Supported architectures (`--arch`)

- **`mamba`** — HF `state-spaces/mamba-*`. Names already match FlowEdge, so this is a
  rename (`backbone.embedding` → `backbone.embeddings`) + normalize + drop-unused
  (`lm_head`, …) pass. `flow.*` tensors, if present, pass through.
