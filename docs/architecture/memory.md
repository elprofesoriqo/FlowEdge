# Memory Arena

One fixed buffer and a bump pointer. That is the whole allocator.

```{mermaid}
%%{init: {'theme':'base','flowchart':{'htmlLabels':false,'nodeSpacing':28,'rankSpacing':34,'useMaxWidth':false},'themeVariables':{'primaryColor':'#f6ead0','primaryBorderColor':'#7b2733','lineColor':'#7b2733','primaryTextColor':'#2b2521','secondaryColor':'#eaddbf','tertiaryColor':'#faf3e2','fontFamily':'system-ui, -apple-system, Segoe UI, Roboto, sans-serif','fontSize':'13px'}}}%%
graph LR
  S["slab"] --> C[cursor]
  C -->|alloc| W[weights]
  C -->|alloc| SC[scratch]
  SC -->|reset_to mark| C
```

At load the engine sizes the slab from the checkpoint, then carves the weights
once, aligned to 64 bytes. Runtime storage is carved from the same slab:
persistent decode state, a persistent resumable-flow workspace, flow scratch,
load-time-transformed SSM matrices, the thread-pool task ring, and the
`std::jthread` objects that own workers.

## How it works

The arena owns one `std::byte` slab and a cursor. `alloc(n, a)` uses
`std::align`, hands back the aligned span, and advances by `n`. Every allocation
is O(1) and contiguous, with no free list and no per-object header.

Scratch is stack-like: `mark()` records the cursor and `reset_to()` winds it back. A layer carves its intermediates, and the caller rewinds to the mark afterward, so the next layer reuses the same bytes.

The slab size is fixed at load:

$$
\text{slab} = W_{bytes} + S_{decode} + S_{flow} + R_{pool} + O_{alignment}
$$

$W_{bytes}$ is the stored weight footprint in bytes (2 bytes per element for
`BF16` weights). $S_{decode}$ covers Mamba conv windows, SSM state, and the
state-major $-\exp(A_{\log})$ matrix per layer. $S_{flow}$ covers the resumable
ODE state, velocity temporaries, and projection scratch. $R_{pool}$
covers the power-of-two task ring and worker objects. `alloc` returns `nullptr`
on exhaustion, so a model that does not fit fails at load instead of drifting
into the control loop.

## Why this way

A general allocator is a source of jitter. `malloc` may walk a free list, take a lock, or fault in a fresh page on first touch, and any of those adds a tail to the step. A bump allocator does none of that: an allocation is a pointer bump, and reuse is a reset back to a mark. The slab itself is faulted in once at load, so every address the hot path touches is already resident.

Multiple engines may retain one immutable `ModelWeights` checkpoint store. Tensor views point into
that reference-counted storage, while each engine keeps its arena, decode state, sampler workspace,
thread-pool structures, and load-time transformed constants private. The shared ownership operation
happens only when an engine is created or destroyed; inference uses raw immutable views.

The 64-byte alignment pays off twice. It matches the cache line, so a buffer never straddles two lines, and it is a multiple of the SIMD width, so the kernels can use aligned loads. Unaligned buffers would cost split loads and cross-line traffic in every kernel.

Rewinding to a mark is a cache decision as much as a bookkeeping one. A layer's intermediates are dead once the layer finishes, so reusing that region keeps the working set small and hot. A 24-layer forward touches one scratch region rather than 24.

The footprint is $O(1)$ in the action horizon, where a growing KV cache would be $O(N)$. An embedded target cannot bound an $O(N)$ cache, but here the peak is known at load and checked, so the engine cannot quietly run past its budget.

Source: `src/core/arena/arena.h`.
