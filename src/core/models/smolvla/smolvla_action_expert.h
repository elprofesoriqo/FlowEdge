#pragma once

#include "arena/arena.h"
#include "loader/safetensors.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace fe {

class ThreadPool;

// Native execution boundary for the trained SmolVLA flow expert.  Image,
// language, and VLM-prefix encoding stay outside this class; callers provide
// a captured VLM K/V cache and this class executes the action suffix without
// allocating in its hot path.
struct SmolVLAActionExpertConfig
{
  std::size_t max_state_dim{};
  std::size_t max_action_dim{};
  std::size_t vlm_width{};
  std::size_t expert_width{};
  std::size_t expert_layers{};
  std::size_t attention_width{};
  std::size_t key_value_width{};
  std::size_t mlp_width{};
};

class SmolVLAActionExpert
{
public:
  SmolVLAActionExpert(std::span<const TensorView> weights, Arena& arena) noexcept;

  [[nodiscard]] bool valid() const noexcept { return valid_; }
  [[nodiscard]] const SmolVLAActionExpertConfig& config() const noexcept { return cfg_; }
  void set_pool(ThreadPool* pool) noexcept { pool_ = pool; }

  // Build the exact SmolVLA suffix embedding for a batch-one action chunk.
  // `noisy_actions` is [chunk_size, max_action_dim], `output` is
  // [chunk_size, expert_width].  The caller owns both buffers.
  [[nodiscard]] bool embed_suffix(std::span<const float> noisy_actions, float timestep,
                                  std::span<float> output) noexcept;

  // Project a padded LeRobot state into the VLM prefix width.
  [[nodiscard]] bool project_state(std::span<const float> state, std::span<float> output) noexcept;

  // Project expert hidden states back to padded action coordinates.
  [[nodiscard]] bool project_actions(std::span<const float> hidden,
                                     std::span<float> output) noexcept;

  // Execute all interleaved expert layers for one action chunk. The caller
  // supplies the VLM's per-layer, RoPE-applied K/V cache in layer-major F32
  // layout [expert_layers, prefix_length, key_value_width] and one validity
  // byte per prefix token. This deliberately keeps image/language encoding
  // outside Core while making captured-encoder parity possible.
  [[nodiscard]] bool run_with_prefix_kv(std::span<const float> suffix,
                                        std::span<const float> prefix_keys,
                                        std::span<const float> prefix_values,
                                        std::span<const std::uint8_t> prefix_mask,
                                        std::span<float> output) noexcept;

  // Allocation-free complete action-expert velocity for one denoising step.
  // It composes suffix embedding, cached-VLM attention, and action projection
  // while retaining the image/language VLM boundary outside Core.
  [[nodiscard]] bool denoise_with_prefix_kv(std::span<const float> noisy_actions, float timestep,
                                            std::span<const float> prefix_keys,
                                            std::span<const float> prefix_values,
                                            std::span<const std::uint8_t> prefix_mask,
                                            std::span<float> output) noexcept;

private:
  [[nodiscard]] std::span<float> scratch(std::size_t count) noexcept
  {
    return arena_->alloc_span<float, kSimdAlign>(count);
  }

  static constexpr std::size_t kMaxExpertLayers = 48uz;
  struct Layer
  {
    const float* input_norm{};
    const float* post_attention_norm{};
    WeightView q_proj{};
    WeightView k_proj{};
    WeightView v_proj{};
    WeightView o_proj{};
    WeightView gate_proj{};
    WeightView up_proj{};
    WeightView down_proj{};
    bool cross_attention{};
  };
  SmolVLAActionExpertConfig cfg_{};
  std::array<Layer, kMaxExpertLayers> layers_{};
  WeightView state_proj_{};
  const float* state_bias_{};
  WeightView action_in_proj_{};
  const float* action_in_bias_{};
  WeightView action_out_proj_{};
  const float* action_out_bias_{};
  WeightView time_mlp_in_{};
  const float* time_mlp_in_bias_{};
  WeightView time_mlp_out_{};
  const float* time_mlp_out_bias_{};
  const float* final_norm_{};
  Arena* arena_{};
  ThreadPool* pool_{};
  bool valid_{};
};

} // namespace fe
