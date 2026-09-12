# FlowEdge ONNX boundary

This optional package runs a fixed-shape ONNX artifact through ONNX Runtime without adding a graph
runtime to FlowEdge Core or Relay. It validates one float32 input/output contract at setup and writes
each result into caller-owned storage. ONNX Runtime may allocate internally; its allocations and
latency are separate from FlowEdge's native zero-hot-path-allocation guarantee.

```bash
python -m pip install -e 'integrations/onnx[runtime]'
```

Use it for policies that cannot yet be converted to FlowEdge's native safetensors layouts. Unsupported
operators fail during ONNX Runtime session creation with an error that names the provider and directs
the user to a native converter or a compatible execution provider.
