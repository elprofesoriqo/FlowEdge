#include "kernels/kernels.h"

#include <cmath>
#include <cstddef>

// identical scalar code across every ISA backend
// so it lives once here and links alongside the selected SIMD file (avx2/neon/scalar).
namespace fe {

void conv1d_causal(std::span<const float> x, std::span<const float> weight,
                   std::span<const float> bias, std::span<float> y, std::size_t channels,
                   std::size_t length, std::size_t kernel) noexcept
{
  for (std::size_t c{0uz}; c < channels; ++c) {
    const float* __restrict__ xc = x.data() + (c * length);
    const float* __restrict__ wc = weight.data() + (c * kernel);
    float* __restrict__ yc = y.data() + (c * length);
    const float bc = bias[c];
    for (std::size_t t{0uz}; t < length; ++t)
      yc[t] = bc;
    for (std::size_t k{0uz}; k < kernel; ++k) {
      const float wk = wc[k];
      const std::size_t start = (kernel - 1uz) - k;
      for (std::size_t t{start}; t < length; ++t)
        yc[t] += wk * xc[t - start];
    }
  }
}

void conv1d_step(std::span<const float> window, std::span<const float> weight,
                 std::span<const float> bias, std::span<float> y, std::size_t channels,
                 std::size_t kernel) noexcept
{
  for (std::size_t c{0uz}; c < channels; ++c) {
    const float* __restrict__ wc = window.data() + (c * kernel);
    const float* __restrict__ kw = weight.data() + (c * kernel);
    float acc = bias[c];
    for (std::size_t k{0uz}; k < kernel; ++k)
      acc += kw[k] * wc[k];
    y[c] = acc;
  }
}

void rmsnorm(std::span<const float> in, std::span<const float> weight, std::span<float> out,
             std::size_t rows, std::size_t dim) noexcept
{
  constexpr float eps = 1e-5F;
  for (std::size_t r{0uz}; r < rows; ++r) {
    const float* __restrict__ ir = in.data() + (r * dim);
    float* __restrict__ orow = out.data() + (r * dim);
    float ss{0.0F};
    for (std::size_t i{0uz}; i < dim; ++i)
      ss += ir[i] * ir[i];
    const float scale = 1.0F / std::sqrt((ss / static_cast<float>(dim)) + eps);
    for (std::size_t i{0uz}; i < dim; ++i)
      orow[i] = ir[i] * scale * weight[i];
  }
}

} // namespace fe
