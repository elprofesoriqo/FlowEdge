#pragma once

#include "arena/arena.h"
#include "loader/safetensors.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace fe {

class ThreadPool;

// Deliberately narrow, fixed-shape causal Transformer contract. It models the
// shared decoder part of policy/VLA architectures, not any vendor model format.
struct TransformerConfig
{
  std::size_t vocab{}, d_model{}, n_layers{}, n_heads{}, max_sequence{}, mlp_dim{};
};

class Transformer
{
public:
  Transformer(std::span<const TensorView> weights, Arena& arena) noexcept;

  [[nodiscard]] bool valid() const noexcept { return valid_; }
  [[nodiscard]] const TransformerConfig& config() const noexcept { return cfg_; }
  [[nodiscard]] std::size_t state_size() const noexcept
  {
    return 2uz * cfg_.n_layers * cfg_.max_sequence * cfg_.d_model;
  }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_pool(ThreadPool* pool) noexcept { pool_ = pool; }
  void reset() noexcept;

  // Full-prefix token inference. It resets and fills the same fixed KV cache
  // used by streaming decode, making prefix/decode parity testable by design.
  [[nodiscard]] bool forward_tokens(std::span<const std::int32_t> tokens,
                                    std::span<float> output) noexcept;
  // Prefill caller-supplied embeddings with an optional prefix mask. A mask is
  // batch-one, byte-valued, and must contain leading ones followed by padding
  // zeros. This is the first explicit condition-sequence boundary for VLA
  // adapters; cross-attention and the SmolVLA expert remain separate work.
  [[nodiscard]] bool forward_embeddings(std::span<const float> embeddings,
                                        std::span<const std::uint8_t> attention_mask,
                                        std::span<float> output) noexcept;
  // Process one already-embedded token. This is the VLA boundary: image,
  // proprioception and language encoders remain outside the generic backbone.
  [[nodiscard]] bool decode_embedding(std::span<const float> embedding,
                                      std::span<float> output) noexcept;
  [[nodiscard]] bool decode_embedding(std::span<const float> embedding,
                                      std::span<const std::uint8_t> attention_mask,
                                      std::span<float> output) noexcept;
  [[nodiscard]] bool decode_token(std::int32_t token, std::span<float> output) noexcept;

private:
  struct Layer
  {
    const float* ln1_weight{};
    const float* ln1_bias{};
    WeightView qkv{};
    const float* qkv_bias{};
    WeightView attention_out{};
    const float* attention_out_bias{};
    const float* ln2_weight{};
    const float* ln2_bias{};
    WeightView mlp_in{};
    const float* mlp_in_bias{};
    WeightView mlp_out{};
    const float* mlp_out_bias{};
    float* key_cache{};   // [max_sequence][d_model]
    float* value_cache{}; // [max_sequence][d_model]
  };

  [[nodiscard]] bool decode_layer(Layer& layer, std::span<float> hidden,
                                  std::span<const std::uint8_t> attention_mask) noexcept;
  [[nodiscard]] std::span<float> scratch(std::size_t count) noexcept
  {
    return arena_->alloc_span<float, kSimdAlign>(count);
  }

  static constexpr std::size_t kMaxLayers = 48uz;
  TransformerConfig cfg_{};
  std::array<Layer, kMaxLayers> layers_{};
  const float* token_embedding_{};
  const float* position_embedding_{};
  const float* final_norm_weight_{};
  const float* final_norm_bias_{};
  Arena* arena_{};
  ThreadPool* pool_{};
  std::size_t position_{};
  bool valid_{};
};

} // namespace fe
