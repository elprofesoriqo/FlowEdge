# Model Porting Analysis

This analysis reflects the public model landscape in August 2026. A model is only an exact port
when its tensor layout and computation are reproduced; sharing a flow-matching objective is not
enough.

| Target | Fit with FlowEdge today | Missing exact-port work | Recommendation |
| --- | --- | --- | --- |
| Custom encoder + FlowEdge MLP flow head | Direct | Train/export `flow.*`; feed one F32 condition vector | Production-ready integration path |
| Mamba-1 language/control checkpoints | Strong backbone fit | LM projection, sampling, tokenizer adapter, checkpoint mapping | Small, useful port |
| Mamba-3 SISO/MIMO | Strong architectural fit, incompatible recurrence | Complex/angle state, trapezoidal update, K/V state, grouped heads, LM head | Highest-value new backbone |
| SmolVLA | Objective fits; architecture does not | Transformer flow expert with self/cross attention over VLM tokens | Port after condition-sequence ABI |
| pi0 / pi0.5 | Objective fits; architecture does not | PaliGemma/VLM token interface and coupled flow action expert | Adapter first, exact port later |
| GR00T N1.5 | Objective fits; architecture does not | DiT action model cross-attending to Eagle VLM embeddings | External runtime first |
| pi0-FAST | Does not use the flow path | FAST action tokenizer and autoregressive transformer decoder | Defer unless token actions become a core goal |

## What works immediately

The head-only checkpoint and `sample_condition` ABI are intentionally model-runtime neutral. A
company can keep a supported VLM or perception encoder in PyTorch, JAX, TensorRT, or ONNX Runtime,
train the compact FlowEdge velocity MLP against its pooled latent, and deploy only that action head
in this engine. This is an integration or distillation target, not a claim that an upstream
checkpoint converts unchanged.

This boundary is useful because current VLA action experts are structurally richer than FlowEdge's
MLP. [SmolVLA](https://huggingface.co/blog/smolvla) uses a transformer flow-matching expert with
interleaved attention. [OpenPI](https://github.com/Physical-Intelligence/openpi) describes pi0 and
pi0.5 as flow-based models, while pi0-FAST is autoregressive. [GR00T N1.5](https://research.nvidia.com/labs/gear/gr00t-n15/)
uses a DiT that cross-attends to VLM embeddings. Their weights cannot be flattened into
`flow.cond_proj` without retraining.

## Recommended exact port: Mamba-3

The next backbone should be Mamba-3, starting with SISO decode and then MIMO. The official
[Mamba repository](https://github.com/state-spaces/mamba) now includes Mamba-3, and the
[paper](https://arxiv.org/abs/2603.15569) describes it as inference-first, with a richer recurrence,
complex-valued state updates, and a MIMO formulation. Those properties align with FlowEdge's fixed
state, streaming-first runtime better than adding a general transformer graph executor.

The implementation should be a new `models/mamba3/` module rather than conditionals in Mamba-1:

1. Add scalar reference kernels for SISO recurrence and step decode.
2. Add AVX2/NEON state-major kernels, keeping angle and recurrent state caller-owned.
3. Define the Mamba-3 state payload and architecture identifier inside the existing versioned,
   checksummed snapshot envelope.
4. Add embedding, normalization, and LM output projection, with top-k/top-p sampling outside the
   deterministic backbone call.
5. Validate token-by-token output and state against the official implementation before optimizing
   chunked prefill.
6. Add MIMO only after SISO correctness; its extra K/V state should have an explicit arena formula.

Mamba-2 remains useful as an intermediate reference, but implementing it first only makes sense if a
specific customer checkpoint requires it. Mamba-3 is the more distinctive long-term ML-systems
target.

## Recommended VLA port boundary

Exact SmolVLA, pi0, and GR00T ports need a condition-sequence ABI, not just a larger vector. A future
version should describe a read-only matrix `[tokens, width]`, optional attention mask, and a lifetime
valid across resumable solver calls. That ABI can serve several action experts without importing the
VLM itself.

SmolVLA is the best first exact VLA expert because it is compact, openly documented, and explicitly
designed for low-latency inference. pi0/pi0.5 should follow through an OpenPI conformance harness.
GR00T should initially remain an external NVIDIA runtime feeding an independently trained FlowEdge
head; reproducing Eagle plus DiT would turn this project into the broad graph runtime it deliberately
avoids.

## System-level implication

Modern VLA deployment is also an execution problem. LeRobot's
[asynchronous inference](https://github.com/huggingface/lerobot/blob/main/docs/source/async.mdx)
decouples action execution from prediction, and its
[Real-Time Chunking](https://huggingface.co/docs/lerobot/main/rtc) documentation addresses pauses
and discontinuities when chunks arrive late. FlowEdge's cooperative steps are the low-level
mechanism; the proposed [FlowEdge Relay](../ecosystem/relay-proposal) is where freshness, overlap,
backpressure, and distributed scheduling belong.
