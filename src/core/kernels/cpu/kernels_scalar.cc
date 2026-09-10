#include "kernels/kernels.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>

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

void gate_silu(std::span<const float> a, std::span<const float> g, std::span<float> out) noexcept
{
  for (std::size_t i{0uz}; i < out.size(); ++i) {
    const float v = g[i];
    out[i] = a[i] * (v / (1.0F + std::exp(-v)));
  }
}

void silu(std::span<float> x) noexcept
{
  for (std::size_t i{0uz}; i < x.size(); ++i)
    x[i] = x[i] / (1.0F + std::exp(-x[i]));
}

void softplus(std::span<float> x) noexcept
{
  for (std::size_t i{0uz}; i < x.size(); ++i) {
    const float v = x[i];
    x[i] = std::fmax(v, 0.0F) + std::log1p(std::exp(-std::fabs(v))); // stable: no overflow for v≫0
  }
}

void matmul(std::span<const float> in, std::span<const float> w, std::span<float> out,
            std::size_t rows, std::size_t in_dim, std::size_t out_dim, ThreadPool* pool) noexcept
{
  auto op = [&](std::size_t lo, std::size_t hi) noexcept {
    for (std::size_t o{lo}; o < hi; ++o) {
      const float* __restrict__ wr = w.data() + (o * in_dim);
      for (std::size_t r{0uz}; r < rows; ++r) {
        const float* __restrict__ ir = in.data() + (r * in_dim);
        float acc{0.0F};
        for (std::size_t i{0uz}; i < in_dim; ++i)
          acc += ir[i] * wr[i];
        out.data()[(r * out_dim) + o] = acc;
      }
    }
  };
  const unsigned tasks = matmul_task_count(pool, rows, in_dim, out_dim, MatmulWeightType::kF32);
  if (pool != nullptr && tasks > 1u)
    parallel_for(*pool, out_dim, tasks, op);
  else
    op(0uz, out_dim);
}

void matmul(std::span<const float> in, std::span<const uint16_t> w, std::span<float> out,
            std::size_t rows, std::size_t in_dim, std::size_t out_dim, ThreadPool* pool) noexcept
{
  auto op = [&](std::size_t lo, std::size_t hi) noexcept {
    for (std::size_t o{lo}; o < hi; ++o) {
      const uint16_t* __restrict__ wr = w.data() + (o * in_dim);
      for (std::size_t r{0uz}; r < rows; ++r) {
        const float* __restrict__ ir = in.data() + (r * in_dim);
        float acc{0.0F};
        for (std::size_t i{0uz}; i < in_dim; ++i)
          acc += ir[i] * std::bit_cast<float>(static_cast<std::uint32_t>(wr[i]) << 16u);
        out.data()[(r * out_dim) + o] = acc;
      }
    }
  };
  const unsigned tasks = matmul_task_count(pool, rows, in_dim, out_dim, MatmulWeightType::kBF16);
  if (pool != nullptr && tasks > 1u)
    parallel_for(*pool, out_dim, tasks, op);
  else
    op(0uz, out_dim);
}

void discretize_and_scan(std::span<const float> delta, std::span<const float> a_neg,
                         std::span<const float> b, std::span<const float> u,
                         std::span<const float> c_proj, std::span<const float> d_skip,
                         std::span<float> h, std::span<float> y, std::size_t length,
                         std::size_t d_inner, std::size_t d_state, bool reset_state,
                         std::size_t row_stride) noexcept
{
  float* __restrict__ hs = h.data();
  row_stride = row_stride == 0uz ? d_state : row_stride;
  if (reset_state)
    std::fill_n(hs, d_inner * d_state, 0.0F);

  for (std::size_t t{0uz}; t < length; ++t) {
    const float* __restrict__ dt = delta.data() + (t * d_inner);
    const float* __restrict__ ut = u.data() + (t * d_inner);
    const float* __restrict__ bt = b.data() + (t * row_stride);
    const float* __restrict__ c_t = c_proj.data() + (t * row_stride);
    const float* __restrict__ dk = d_skip.data();
    float* __restrict__ y_t = y.data() + (t * d_inner);

    for (std::size_t c{0uz}; c < d_inner; ++c)
      y_t[c] = dk[c] * ut[c]; // skip term

    for (std::size_t n{0uz}; n < d_state; ++n) {
      float* __restrict__ hn = hs + (n * d_inner);
      const float* __restrict__ an = a_neg.data() + (n * d_inner);
      const float bn = bt[n];
      const float cn = c_t[n];
      for (std::size_t c{0uz}; c < d_inner; ++c) {
        const float da_nc = std::exp(dt[c] * an[c]);
        const float dbu_nc = dt[c] * bn * ut[c];
        hn[c] = (da_nc * hn[c]) + dbu_nc;
        y_t[c] += hn[c] * cn;
      }
    }
  }
}

} // namespace fe
