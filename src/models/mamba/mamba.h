#pragma once

#include "arena/arena.h"
#include "loader/safetensors.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace fe {

class ThreadPool;

struct MambaConfig
{
  std::size_t d_model{}, d_inner{}, d_state{}, d_conv{}, dt_rank{}, n_layers{}, vocab{};
};

class Mamba
{
public:
  Mamba(std::span<const TensorView> weights, Arena& scratch) noexcept;

  [[nodiscard]] bool valid() const noexcept { return ok_; }
  [[nodiscard]] const MambaConfig& config() const noexcept { return cfg_; }
  [[nodiscard]] const float* embedding() const noexcept { return emb_; }

  void set_pool(ThreadPool* p, std::span<Arena> arenas) noexcept
  {
    pool_ = p;
    worker_arenas_ = arenas;
  }

  // [seq_len][d_model]
  void forward(std::span<const float> input, std::span<float> output, std::size_t seq_len) noexcept;

  // persistent decode state (conv window + SSM h) per layer, in floats
  [[nodiscard]] std::size_t state_size() const noexcept
  {
    return cfg_.n_layers * cfg_.d_inner * (cfg_.d_conv + cfg_.d_state);
  }

  // advance one token: x[d_model] in, out[d_model] out
  // `state` carries SSM+conv across calls
  void decode(std::span<const float> x, std::span<float> state, std::span<float> out) noexcept;

private:
  struct Layer
  {
    const float* norm;
    const uint16_t* in_proj;
    const float* conv_w;
    const float* conv_b;
    const uint16_t* x_proj;
    const uint16_t* dt_w;
    const float* dt_b;
    const float* a_log;
    const float* d;
    const uint16_t* out_proj;
  };

  void layer_forward(const Layer& lw, std::span<float> hidden, std::size_t seq_len) noexcept;
  void decode_layer(const Layer& lw, std::span<float> hidden, std::span<float> lstate) noexcept;

  // 64B-aligned scratch span carved from the arena
  [[nodiscard]] std::span<float> arena_span(std::size_t n) noexcept
  {
    return scratch_->alloc_span<float>(n, kSimdAlign);
  }

  static constexpr std::size_t kMaxLayers = 64uz;
  MambaConfig cfg_{};
  std::array<Layer, kMaxLayers> layers_{};
  const float* emb_{};
  const float* norm_f_{};
  Arena* scratch_{};
  ThreadPool* pool_{nullptr};
  std::span<Arena> worker_arenas_{};
  bool ok_{false};
};

} // namespace fe