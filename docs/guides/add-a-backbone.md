# Add a Backbone

A backbone reads tokens and returns a conditioning vector. It also runs one-token streaming decode with fixed state. Follow Mamba.

## The contract

```cpp
class MyBackbone {
  MyBackbone(std::span<const TensorView> weights, Arena& scratch) noexcept;
  bool valid() const noexcept;
  const Config& config() const noexcept;      // d_model, n_layers, vocab, ...
  const float* embedding() const noexcept;    // [vocab][d_model], the engine gathers tokens
  std::size_t state_size() const noexcept;    // floats of streaming state
  void forward(std::span<const float> in, std::span<float> out, std::size_t seq_len) noexcept;
  void decode(std::span<const float> x, std::span<float> state, std::span<float> out) noexcept;
};
```

The conditioning vector is the last row of `forward`, size `d_model`.

## Steps

1. Create `src/core/models/<name>/`. Derive config from checkpoint shapes. Store raw weight pointers.
2. Implement `forward` for the prefix. Carve per-layer scratch from the arena and rewind per layer.
3. Implement `decode` for one token. Keep all state fixed-size and in the caller `state` span. It must equal `forward` step by step.
4. Add kernels behind `kernels.h`. Do not grow state with sequence length. A transformer sizes its KV-cache to the max prefix.
5. Add a converter mapping. Select the backbone at load from the tensor names.
6. Verify against PyTorch. Check `decode` matches `forward`. Add tests and an ADR.

Reference: `src/core/models/mamba/`.
