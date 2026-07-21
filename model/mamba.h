#pragma once

#include "../arena/arena.h"
#include "../loader/safetensors.h"

#include <array>
#include <cstddef>
#include <span>

namespace fe {

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

  // [seq_len][d_model]
  void forward(std::span<const float> input, std::span<float> output, std::size_t seq_len) noexcept;

private:
  struct Layer
  {
    const float* norm;
    const float* in_proj;
    const float* conv_w;
    const float* conv_b;
    const float* x_proj;
    const float* dt_w;
    const float* dt_b;
    const float* a_log;
    const float* d;
    const float* out_proj;
  };

  void layer_forward(const Layer& lw, std::span<float> hidden, std::size_t seq_len) noexcept;

  static constexpr std::size_t kMaxLayers = 64uz;
  MambaConfig cfg_{};
  std::array<Layer, kMaxLayers> layers_{};
  const float* emb_{};
  const float* norm_f_{};
  Arena* scratch_{};
  bool ok_{false};
};

} // namespace fe