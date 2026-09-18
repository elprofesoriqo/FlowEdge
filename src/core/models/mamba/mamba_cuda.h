#pragma once

#include <array>
#include <cstddef>
#include <span>

namespace fe {

// Persistent device copies of backbone.* weights and Mamba scratch. Load allocates;
// forward/decode must not cudaMalloc. Host decode state still round-trips.
class CudaMambaResident
{
public:
  static constexpr std::size_t kMaxLayers = 64;
  static constexpr std::size_t kMaxSeq = 512;

  struct LayerHost
  {
    const float* norm{};
    const float* in_proj{};
    const float* conv_w{};
    const float* conv_b{};
    const float* x_proj{};
    const float* dt_w{};
    const float* dt_b{};
    const float* a_neg{};
    const float* d{};
    const float* out_proj{};
  };

  CudaMambaResident() = default;
  CudaMambaResident(const CudaMambaResident&) = delete;
  CudaMambaResident(CudaMambaResident&&) = delete;
  CudaMambaResident& operator=(const CudaMambaResident&) = delete;
  CudaMambaResident& operator=(CudaMambaResident&&) = delete;

  void release() noexcept;

  [[nodiscard]] bool load(std::size_t n_layers, std::size_t d_model, std::size_t d_inner,
                          std::size_t d_state, std::size_t d_conv, std::size_t dt_rank,
                          const float* norm_f, const std::array<LayerHost, kMaxLayers>& layers,
                          std::size_t max_seq = kMaxSeq) noexcept;
  [[nodiscard]] bool forward(std::span<const float> input, std::span<float> output,
                             std::size_t seq_len) noexcept;
  [[nodiscard]] bool decode(std::span<const float> x, std::span<float> state,
                            std::span<float> out) noexcept;

private:
  struct DeviceLayer
  {
    float* norm{};
    float* in_proj{};
    float* conv_w{};
    float* conv_b{};
    float* x_proj{};
    float* dt_w{};
    float* dt_b{};
    float* a_neg{};
    float* d{};
    float* out_proj{};
  };

  [[nodiscard]] bool upload(const float* host, std::size_t count, float*& dst) noexcept;
  [[nodiscard]] float* device_floats(std::size_t count) noexcept;
  void layer_forward(const DeviceLayer& layer, std::size_t seq_len) noexcept;
  void decode_layer(const DeviceLayer& layer, float* conv_win, float* h) noexcept;

  std::size_t n_layers_{};
  std::size_t d_model_{};
  std::size_t d_inner_{};
  std::size_t d_state_{};
  std::size_t d_conv_{};
  std::size_t dt_rank_{};
  std::size_t wd_{};
  std::size_t max_seq_{};
  float* norm_f_{};
  std::array<DeviceLayer, kMaxLayers> layers_{};
  float* hidden_{};
  float* normed_{};
  float* xz_{};
  float* z_{};
  float* x_cm_{};
  float* x_conv_{};
  float* x_sm_{};
  float* dbl_{};
  float* dt_in_{};
  float* dt_{};
  float* h_{};
  float* yv_{};
  float* out_{};
  float* state_{};
};

} // namespace fe
