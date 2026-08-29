# Add a Head

A head consumes a conditioning vector and returns an action. It is one module under `src/core/heads/`. Follow the flow head.

## The contract

```cpp
class MyHead {
  MyHead(std::span<const TensorView> weights, Arena& scratch) noexcept;
  bool valid() const noexcept;
  void sample(std::span<const float> cond,    // conditioning vector, size d_model
              std::span<const float> noise,   // action_dim
              std::size_t steps, Method m,
              std::span<float> out) noexcept;  // action chunk
};
```

The constructor resolves weight pointers by name and derives config from their shapes. `sample` reads the conditioning vector and writes the action. It allocates nothing.

## Steps

1. Create `src/core/heads/<name>/<name>.{h,cc}`. Resolve weight pointers in the constructor. Set an `ok_` flag only when every required tensor is present.
2. Add any missing kernel to `kernels.h`. If it is public, implement it in
   `kernels_avx2.cc`, `kernels_neon.cc`, and `kernels_scalar.cc`, or the
   fallback build will not link.
3. Carve scratch with `arena_span`. Take `mark()` at the top of `sample` and `reset_to` at the end. Nothing allocates on the hot path.
4. Add a converter mapping so a checkpoint loads. See [Converter](converter). Carry the normalization stats and un-normalize the action if the checkpoint stores them.
5. Add a sampling entrypoint to the C-ABI, or a head selector on `fe_engine_sample`.
6. Add a PyTorch reference to `scripts/torch_ref.py` and wire the [ULP gate](verification).
7. Add tests to `test/tests.cc`. Add an example. Tick the head on the home page cards and write an ADR.

Reference: `src/core/heads/flow/flow.h` and `flow.cc`.
