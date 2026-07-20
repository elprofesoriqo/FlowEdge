#include "kernels.h"

#include <cmath>
#include <cstddef>
#include <numbers>
#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace fe {
namespace {

#if defined(__AVX2__) || defined(__ARM_NEON)
constexpr float exp_clamp = 88.3762F;
constexpr float log2e = std::numbers::log2e_v<float>;
constexpr float ln2_hi = 0.693359375F; // Cephes two-part ln2
constexpr float ln2_lo = -2.12194440e-4F;
constexpr float exp_p0 = 1.9875691500e-4F;
constexpr float exp_p1 = 1.3981999507e-3F;
constexpr float exp_p2 = 8.3334519073e-3F;
constexpr float exp_p3 = 4.1665795894e-2F;
constexpr float exp_p4 = 1.6666665459e-1F;
constexpr float exp_p5 = 5.0000001201e-1F;
#endif

#if defined(__AVX2__)
// Cephes 8-wide expf (~1 ULP)
// polynomial back into a per-element libm exp() call
__m256 exp8(__m256 x) noexcept
{
  x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-exp_clamp)), _mm256_set1_ps(exp_clamp));
  const __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(log2e)),
                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(ln2_hi), x);
  x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(ln2_lo), x);
  __m256 y = _mm256_set1_ps(exp_p0);
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(exp_p1));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(exp_p2));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(exp_p3));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(exp_p4));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(exp_p5));
  y = _mm256_fmadd_ps(y, _mm256_mul_ps(x, x), x);
  y = _mm256_add_ps(y, _mm256_set1_ps(1.0F));
  const __m256i pow2 =
      _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(fx), _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(y, _mm256_castsi256_ps(pow2));
}
#elif defined(__ARM_NEON)
// same Cephes expf, 4-wide NEON
float32x4_t exp4(float32x4_t x) noexcept
{
  x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-exp_clamp)), vdupq_n_f32(exp_clamp));
  const float32x4_t fx = vrndnq_f32(vmulq_n_f32(x, log2e));
  x = vfmsq_f32(x, fx, vdupq_n_f32(ln2_hi)); // a - b·c
  x = vfmsq_f32(x, fx, vdupq_n_f32(ln2_lo));
  float32x4_t y = vdupq_n_f32(exp_p0);
  y = vfmaq_f32(vdupq_n_f32(exp_p1), y, x); // c + a·b
  y = vfmaq_f32(vdupq_n_f32(exp_p2), y, x);
  y = vfmaq_f32(vdupq_n_f32(exp_p3), y, x);
  y = vfmaq_f32(vdupq_n_f32(exp_p4), y, x);
  y = vfmaq_f32(vdupq_n_f32(exp_p5), y, x);
  y = vfmaq_f32(x, y, vmulq_f32(x, x));
  y = vaddq_f32(y, vdupq_n_f32(1.0F));
  const int32x4_t pow2 = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(fx), vdupq_n_s32(127)), 23);
  return vmulq_f32(y, vreinterpretq_f32_s32(pow2));
}
#endif

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
  std::size_t i{0uz};
#if defined(__AVX2__)
  const __m256 one = _mm256_set1_ps(1.0F);
  const __m256 zero = _mm256_setzero_ps();
  for (; i + 8uz <= out.size(); i += 8uz) {
    const __m256 vg = _mm256_loadu_ps(g.data() + i);
    const __m256 sig = _mm256_div_ps(vg, _mm256_add_ps(one, exp8(_mm256_sub_ps(zero, vg))));
    _mm256_storeu_ps(out.data() + i, _mm256_mul_ps(_mm256_loadu_ps(a.data() + i), sig));
  }
#elif defined(__ARM_NEON)
  const float32x4_t one = vdupq_n_f32(1.0F);
  for (; i + 4uz <= out.size(); i += 4uz) {
    const float32x4_t vg = vld1q_f32(g.data() + i);
    const float32x4_t sig = vdivq_f32(vg, vaddq_f32(one, exp4(vnegq_f32(vg))));
    vst1q_f32(out.data() + i, vmulq_f32(vld1q_f32(a.data() + i), sig));
  }
#endif
  for (; i < out.size(); ++i) {
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
  const float* __restrict__ d = d_skip.data();
  for (std::size_t t{0uz}; t < length; ++t) {
    const float* __restrict__ da_t = delta_a.data() + (t * plane);
    const float* __restrict__ dbu_t = delta_bu.data() + (t * plane);
    const float* __restrict__ c_t = c_proj.data() + (t * d_state);
    const float* __restrict__ u_t = u.data() + (t * d_inner);
    float* __restrict__ y_t = y.data() + (t * d_inner);

    for (std::size_t c{0uz}; c < d_inner; ++c)
      y_t[c] = d[c] * u_t[c]; // skip term, scan accumulates onto it

    // state-major [n][c]
    // ILP and streams at the memory-BW ceiling
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
}

void silu(std::span<float> x) noexcept
{
  for (float& v : x)
    v = v / (1.0F + std::exp(-v));
}

void softplus(std::span<float> x) noexcept
{
  for (float& v : x)
    v = std::log1p(std::exp(v));
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
  for (std::size_t r{0uz}; r < rows; ++r) {
    const float* __restrict__ ir = in.data() + (r * in_dim);
    float* __restrict__ orow = out.data() + (r * out_dim);
    for (std::size_t o{0uz}; o < out_dim; ++o) {
      const float* __restrict__ wr = w.data() + (o * in_dim);
      float acc{0.0F};
      for (std::size_t i{0uz}; i < in_dim; ++i)
        acc += ir[i] * wr[i];
      orow[o] = acc;
    }
  }
}

} // namespace fe
