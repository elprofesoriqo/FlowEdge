# Python

Built with pybind11. Install with pip.

```bash
pip install .
```

```python
import numpy as np, flowedge

e = flowedge.Engine("models/mamba_flow.safetensors")
print(e.action_dim, e.condition_dim, e.d_model, e.thread_count)

tokens = np.array([1, 2, 3, 4], dtype=np.int32)
hidden = e.run(tokens)
noise = np.zeros(e.action_dim, dtype=np.float32)
action = e.sample(prefix=tokens,
                  noise=noise,
                  steps=10, method="euler")
```

The convenience calls accept array-like inputs, normalize their dtype/layout when necessary, and
allocate returned NumPy arrays. A real-time loop can instead own every output and use the zero-copy
forms; those inputs and outputs must expose C-contiguous `int32` or `float32` buffers:

```python
hidden = np.empty((tokens.size, e.d_model), dtype=np.float32)
e.run_into(tokens, hidden)

current = np.empty(e.d_model, dtype=np.float32)
e.step_into(7, current)

action = np.empty(e.action_dim, dtype=np.float32)
e.sample_into(tokens, noise, action, steps=10, method="euler")
```

All inference calls release the GIL while C++ runs.

Pass `threads=0..8` to override the automatic worker pool for a specific engine. If omitted,
`FLOWEDGE_THREADS` is honored and then the automatic default is used:

```python
single_threaded = flowedge.Engine("models/mamba_flow.safetensors", threads=0)
```

## External encoders and cooperative solving

A head-only checkpoint accepts a condition vector produced by PyTorch, TensorRT, ONNX Runtime, a
shared-memory camera process, or another model server. Output is caller-owned:

```python
condition = encoder(observation).astype(np.float32, copy=False)
noise = np.zeros(e.action_dim, dtype=np.float32)
action = np.empty(e.action_dim, dtype=np.float32)

e.sample_condition(condition, noise, action, 8, "heun")
```

The same solve can be split across scheduler quanta without changing its result:

```python
e.flow_begin(condition, noise, 8, "heun")
remaining = e.flow_advance(action, 2)
while remaining:
    remaining = e.flow_advance(action, 1)
```

Only publish `action` as final when `remaining == 0`. One engine owns one resumable solve.

## Streaming state

`step(token)` advances the Mamba recurrence, `reset()` starts a fresh stream, and
`decode_state()` / `restore_decode_state(bytes)` create deterministic branches. Snapshots are
valid only for an engine loaded from the same checkpoint.

Source: `python/flowedge_ext.cc`.
