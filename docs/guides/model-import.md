# Model import paths

| Artifact | Path | Runtime | Allocation contract | Status |
|---|---|---|---|---|
| Mamba / supported flow checkpoint | `convert/convert.py` → `.safetensors` | FlowEdge Core | Zero after setup | Supported |
| LeRobot `diffusion_pusht` directory | `policy_inspect.py` → converter → `flowedge-lerobot` | Core + companion | Zero in native inference after setup | Supported |
| Fixed-shape ONNX policy | `flowedge-onnx` | Optional ONNX Runtime | External runtime; measure separately | Adapter preview |
| ExecuTorch `.pte` | Future companion adapter | ExecuTorch | External runtime; measure separately | Planned |

Use the smallest path that preserves your policy. Native conversion is the only path covered by
FlowEdge's zero-hot-path-allocation contract. The ONNX adapter validates one fixed-shape float32
input/output pair at setup, has no dependency from Core or Relay, and reports operator/provider
failures instead of silently falling back.

```bash
python convert/policy_inspect.py policy_dir --json
python -m pip install -e 'integrations/onnx[runtime]'
```

For an ONNX model, run a deterministic fixture through its source framework and `OnnxAdapter.run_into`;
compare the caller-owned output before using it in a control loop. Dynamic shapes, multiple I/O, and
unsupported operators need a model-specific adapter or native converter rather than a hidden graph runtime.
