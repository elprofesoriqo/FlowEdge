# Comparison and Performance

## How FlowEdge compares

FlowEdge is not a general inference framework. It runs a fixed set of policy architectures as small and as fast as it can. That focus is what separates it from the tools it tends to get measured against.

- **PyTorch.** Built for training and general inference with dynamic model execution. FlowEdge removes
  interpreter and hot-path allocation overhead for its supported fixed models. The trade is a much
  narrower architecture set; run the matched reference script before claiming a speedup.
- **ONNX Runtime.** A large framework that ingests an arbitrary graph and carries heavy dependencies. FlowEdge compiles to a small static binary and allocates nothing on the hot path, but it does not read arbitrary graphs. See [ADR 0005](decisions/0005-no-graph-runtime).
- **ggml and llama.cpp.** Tuned for autoregressive text generation. FlowEdge targets continuous control instead: it integrates a flow-matching ODE over an action chunk rather than decoding one token at a time.

## Performance

The current reference covers Core latency, Relay end-to-end latency and throughput, worker scaling,
shared-weight memory, and cooperative migration overhead on native Windows and WSL. See [Current
performance reference](performance) for the measured environment, complete tables, limitations, and
commands.

FlowEdge does not currently publish a universal PyTorch speedup. `scripts/torch_ref.py` provides the
matched reference paths, but any comparison must use the same host, checkpoint, input, solver, thread
count, warmup, and sustained conditions.
