# Memory Arena

One fixed buffer and a bump pointer.

```{image} ../_static/figures/arena.svg
:alt: Arena slab with weights, decode/ODE, scratch, thread pool
:class: fe-fig
```

At load the engine sizes the slab from the checkpoint, then carves weights
once, 64-byte aligned. Decode state, resumable-flow workspace, SSM matrices,
the task ring, and worker `std::jthread` objects live in the same slab.

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

`malloc` walks lists, takes locks, and faults pages. A bump + `reset_to(mark)` does none of that; the slab is touched once at load. 64-byte alignment matches the cache line and SIMD. One scratch region is reused every layer. Footprint is $O(1)$ in the horizon, not a growing KV cache.

Engines may share immutable `ModelWeights`. Inference uses raw views.

Source: `src/core/arena/arena.h`.
