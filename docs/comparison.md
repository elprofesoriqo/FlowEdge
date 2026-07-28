# Comparison and Performance

## How FlowEdge compares

FlowEdge is not a general inference framework. It runs a fixed set of policy architectures as small and as fast as it can. That focus is what separates it from the tools it tends to get measured against.

- **PyTorch.** Built for training and general inference, with an interpreter over every op and a dynamic allocator. FlowEdge is faster on small control models because it has neither and its memory plan is static. The trade is that it only runs the architectures it implements.
- **ONNX Runtime.** A large framework that ingests an arbitrary graph and carries heavy dependencies. FlowEdge compiles to a small static binary and allocates nothing on the hot path, but it does not read arbitrary graphs. See [ADR 0005](decisions/0005-no-graph-runtime).
- **ggml and llama.cpp.** Tuned for autoregressive text generation. FlowEdge targets continuous control instead: it integrates a flow-matching ODE over an action chunk rather than decoding one token at a time.

## Performance

Mamba-130M forward pass, single-threaded CPU, FP32, prefix length 4.

| Benchmark | Backend | PyTorch | FlowEdge | Speedup |
|---|---|---|---|---|
| `BM_engine_forward` | CPU | 73.0 ms | 28.0 ms | ~2.6x |
| `BM_engine_forward` | CUDA | planned | planned | |
| `BM_engine_forward` | Tenstorrent | planned | planned | |

Regenerate the numbers for your own host and model with `./scripts/bench.sh`.