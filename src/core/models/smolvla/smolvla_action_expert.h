#pragma once

#include "arena/arena.h"
#include "loader/safetensors.h"

#include <cstddef>
#include <span>

namespace fe {

class ThreadPool;

// The projection boundary shared by the trained SmolVLA flow expert.  The
// VLM encoder and the interleaved attention layers are deliberately outside
// this class for now; callers provide the VLM prefix and consume the suffix
// embeddings through a stable, allocation-free interface.
struct SmolVLAActionExpertConfig
{
  std::size_t max_state_dim{};
  std::size_t max_action_dim{};
  std::size_t vlm_width{};
  std::size_t expert_width{};
  std::size_t expert_layers{};
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

private:
  [[nodiscard]] std::span<float> scratch(std::size_t count) noexcept
  {
    return arena_->alloc_span<float, kSimdAlign>(count);
  }

  static constexpr std::size_t kMaxExpertLayers = 48uz;
  SmolVLAActionExpertConfig cfg_{};
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
  Arena* arena_{};
  ThreadPool* pool_{};
  bool valid_{};
};

} // namespace fe
