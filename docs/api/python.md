# Python

Built with pybind11. Install with pip.

```bash
pip install .
```

```python
import numpy as np, flowedge

e = flowedge.Engine("models/mamba_flow.safetensors")
print(e.action_dim, e.d_model)

hidden = e.run(tokens=[1, 2, 3, 4])
action = e.sample(prefix=[1, 2, 3, 4],
                  noise=np.zeros(e.action_dim, dtype="float32"),
                  steps=10, method="euler")
```

The module covers loading, `run` and `sample`. Streaming decode, `fe_engine_step` and `fe_engine_reset`, is C-ABI only. Source: `python/flowedge_ext.cc`.
