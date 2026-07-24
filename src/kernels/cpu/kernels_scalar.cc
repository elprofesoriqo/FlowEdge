#include "kernels/kernels.h"

#include <cmath>
#include <cstddef>

namespace fe {
namespace {

// 1 recurrence step, state-major [n][c]: advance h in place, emit y (skip term + sum_n h·c)
inline void scan_advance(const float* __restrict__ da_t, const float* __restrict__ dbu_t,
                         const float* __restrict__ c_t, const float* __restrict__ u_t,
                         const float* __restrict__ d, float* __restrict__ hs,
                         float* __restrict__ y_t, std::size_t d_inner, std::size_t d_state) noexcept
{
  for (std::size_t c{0uz}; c < d_inner; ++c)
    y_t[c] = d[c] * u_t[c]; // skip term, scan accumulates onto it
  for (std::size_t n{0uz}; n < d_state; ++n) {
    float* __restrict__ hn = hs + (n * d_inner);
    const float* __restrict__ da_n = da_t + (n * d_inner);
    const float* __restrict__ dbu_n = dbu_t + (n * d_inner);
    const float cn = c_t[n];
    for (std::size_t c{0uz}; c < d_inner; ++c) {
      hn[c] = (da_n[c] * hn[c]) + dbu_n[c];
      y_t[c] += hn[c] * cn;
    }
  }
}

} // namespace

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

void gate_silu(std::span<const float> a, std::span<const float> g, std::span<float> out) noexcept
{
  for (std::size_t i{0uz}; i < out.size(); ++i) {
    const float v = g[i];
    out[i] = a[i] * (v / (1.0F + std::exp(-v)));
  }
}

void selective_scan(std::span<const float> delta_a, std::span<const float> delta_bu,
                    std::span<const float> c_proj, std::span<const float> d_skip,
                    std::span<const float> u, std::span<float> h, std::span<float> y,
                    std::size_t length, std::size_t d_inner, std::size_t d_state) noexcept
{
  float* __restrict__ hs = h.data();
  for (std::size_t i{0uz}; i < d_inner * d_state; ++i)
    hs[i] = 0.0F; // h_0 = 0
  const std::size_t plane = d_state * d_inner;
  for (std::size_t t{0uz}; t < length; ++t)
    scan_advance(delta_a.data() + (t * plane), delta_bu.data() + (t * plane),
                 c_proj.data() + (t * d_state), u.data() + (t * d_inner), d_skip.data(), hs,
                 y.data() + (t * d_inner), d_inner, d_state);
}

void scan_step(std::span<const float> delta_a, std::span<const float> delta_bu,
               std::span<const float> c_proj, std::span<const float> d_skip,
               std::span<const float> u, std::span<float> h, std::span<float> y,
               std::size_t d_inner, std::size_t d_state) noexcept
{
  scan_advance(delta_a.data(), delta_bu.data(), c_proj.data(), u.data(), d_skip.data(), h.data(),
               y.data(), d_inner, d_state); // h persists across calls
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

void silu(std::span<float> x) noexcept
{
  for (std::size_t i{0uz}; i < x.size(); ++i)
    x[i] = x[i] / (1.0F + std::exp(-x[i]));
}

void softplus(std::span<float> x) noexcept
{
  for (std::size_t i{0uz}; i < x.size(); ++i)
    x[i] = std::log1p(std::exp(x[i]));
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

void matmul(std::span<const float> in, std::span<const float> w, std::span<float> out,
            std::size_t rows, std::size_t in_dim, std::size_t out_dim) noexcept
{
  for (std::size_t o{0uz}; o < out_dim; ++o) {
    const float* __restrict__ wr = w.data() + (o * in_dim);
    for (std::size_t r{0uz}; r < rows; ++r) {
      const float* __restrict__ ir = in.data() + (r * in_dim);
      float acc{0.0F};
      for (std::size_t i{0uz}; i < in_dim; ++i)
        acc += ir[i] * wr[i];
      out.data()[(r * out_dim) + o] = acc;
    }
  }
}

void discretize(std::span<const float> delta, std::span<const float> a_log,
                std::span<const float> b, std::span<const float> u, std::span<float> delta_a,
                std::span<float> delta_bu, std::span<float> a_work, std::size_t length,
                std::size_t d_inner, std::size_t d_state) noexcept
{
  const std::size_t plane = d_inner * d_state;
  // A = -exp(a_log): compute exp then transpose + negate into a_work[n][c].
  float* __restrict__ tmp = delta_a.data();
  for (std::size_t i{0uz}; i < plane; ++i)
    tmp[i] = std::exp(a_log[i]);

  float* __restrict__ a_sm = a_work.data();
  for (std::size_t c{0uz}; c < d_inner; ++c)
    for (std::size_t n{0uz}; n < d_state; ++n)
      a_sm[(n * d_inner) + c] = -tmp[(c * d_state) + n];

  // delta_a[t,n,c] = exp(delta[t,c]·A[n,c])
  // delta_bu[t,n,c] = delta[t,c]·b[t,n]·u[t,c]
  for (std::size_t t{0uz}; t < length; ++t) {
    const float* __restrict__ dt = delta.data() + (t * d_inner);
    const float* __restrict__ ut = u.data() + (t * d_inner);
    const float* __restrict__ bt = b.data() + (t * d_state);
    for (std::size_t n{0uz}; n < d_state; ++n) {
      const float* __restrict__ an = a_sm + (n * d_inner);
      float* __restrict__ da_n = delta_a.data() + (((t * d_state) + n) * d_inner);
      float* __restrict__ dbu_n = delta_bu.data() + (((t * d_state) + n) * d_inner);
      const float bn = bt[n];
      for (std::size_t c{0uz}; c < d_inner; ++c) {
        da_n[c] = std::exp(dt[c] * an[c]);
        dbu_n[c] = dt[c] * bn * ut[c];
      }
    }
  }
}

} // namespace fe
