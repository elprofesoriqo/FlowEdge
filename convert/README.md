# convert — checkpoint → FlowEdge `.safetensors`

FlowEdge loads fixed tensor layouts: `backbone.*` for the SSM, `flow.*` for the
flow-matching action head, and `dp.*` for the Diffusion Policy action head. This
tool maps supported PyTorch / Hugging Face checkpoints into those layouts.

```bash
pip install torch safetensors huggingface_hub
python convert/convert.py <source> models/my_model.safetensors \
  [--arch mamba|diffusion] [--dtype f32|bf16]
```

`<source>` may be a `.safetensors`, `.pt`, `.pth`, or `.bin` state dict. For a
Diffusion Policy it may also be the downloaded model directory.

## Supported architectures

- **`mamba`** — Hugging Face `state-spaces/mamba-*`. Names already match
  FlowEdge, so conversion normalizes the embedding key and drops unused state.
  `flow.*` tensors, if present, pass through.

- **`diffusion`** — the LeRobot `diffusion_pusht` `ConditionalUnet1D`. The
  converter finds `config.json` beside the model or accepts `--config`. It keeps
  the action U-Net and MIN_MAX action statistics and deliberately drops the
  ResNet image encoder. Runtime input is the flattened observation condition
  immediately before LeRobot's U-Net (132 float values for the reference
  checkpoint). Modern model directories with `policy_postprocessor.json` are
  detected automatically; their `action.min` and `action.max` tensors are read
  from the referenced processor state file. Use `--processor` when the sidecar
  has a non-standard filename.

```bash
hf download lerobot/diffusion_pusht --revision 84a7c23178445c6bbf7e1a884ff497017910f653 \
  --local-dir models/diffusion_pusht
python convert/convert.py models/diffusion_pusht \
  models/diffusion_pusht.flowedge.safetensors --arch diffusion --dtype f32
```

The initial diffusion path supports `squaredcos_cap_v2`, epsilon prediction,
FiLM scale modulation, GroupNorm, fixed horizons, and MIN_MAX action
normalization. Legacy embedded statistics and modern processor sidecars are
supported; converted diffusion checkpoints carry a deployment profile so
`flowedge-inspect` can validate the action contract before deployment. Other
normalization modes fail conversion with a targeted error.

Validate a downloaded LeRobot policy directory before conversion:

```bash
python convert/policy_inspect.py models/diffusion_pusht --json
```
