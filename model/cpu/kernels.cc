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
constexpr float ln2_hi = 0.693359375F;
constexpr float ln2_lo = -2.12194440e-4F;
constexpr float exp_p0 = 1.9875691500e-4F;
constexpr float exp_p1 = 1.3981999507e-3F;
constexpr float exp_p2 = 8.3334519073e-3F;
constexpr float exp_p3 = 4.1665795894e-2F;
constexpr float exp_p4 = 1.6666665459e-1F;
constexpr float exp_p5 = 5.0000001201e-1F;
constexpr float sqrt_half = 0.707106781F; // Cephes logf mantissa split point
constexpr float log_p0 = 7.0376836292e-2F;
constexpr float log_p1 = -1.1514610310e-1F;
constexpr float log_p2 = 1.1676998740e-1F;
constexpr float log_p3 = -1.2420140846e-1F;
constexpr float log_p4 = 1.4249322787e-1F;
constexpr float log_p5 = -1.6668057665e-1F;
constexpr float log_p6 = 2.0000714765e-1F;
constexpr float log_p7 = -2.4999993993e-1F;
constexpr float log_p8 = 3.3333331174e-1F;
#endif

#if defined(__AVX2__)
// Cephes 8-wide expf (~1 ULP)
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
// Cephes 8-wide logf (~1 ULP)
__m256 log8(__m256 x) noexcept
{
  __m256 e = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_srli_epi32(_mm256_castps_si256(x), 23),
                                                 _mm256_set1_epi32(0x7E))); // unbiased exp
  x = _mm256_or_ps(_mm256_and_ps(x, _mm256_castsi256_ps(
                                        _mm256_set1_epi32(static_cast<int>(0x807FFFFFU)))),
                   _mm256_set1_ps(0.5F)); // mantissa in [0.5,1)
  const __m256 lo = _mm256_cmp_ps(x, _mm256_set1_ps(sqrt_half), _CMP_LT_OS);
  e = _mm256_sub_ps(e, _mm256_and_ps(_mm256_set1_ps(1.0F), lo));
  x = _mm256_sub_ps(_mm256_add_ps(x, _mm256_and_ps(x, lo)), _mm256_set1_ps(1.0F));
  const __m256 z = _mm256_mul_ps(x, x);
  __m256 y = _mm256_set1_ps(log_p0);
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(log_p1));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(log_p2));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(log_p3));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(log_p4));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(log_p5));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(log_p6));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(log_p7));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(log_p8));
  y = _mm256_mul_ps(_mm256_mul_ps(y, x), z);
  y = _mm256_fmadd_ps(e, _mm256_set1_ps(ln2_lo), y);
  y = _mm256_fnmadd_ps(z, _mm256_set1_ps(0.5F), y);
  x = _mm256_add_ps(x, y);
  return _mm256_fmadd_ps(e, _mm256_set1_ps(ln2_hi), x);
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
// same Cephes logf, 4-wide NEON
float32x4_t log4(float32x4_t x) noexcept
{
  float32x4_t e =
      vcvtq_f32_s32(vsubq_s32(vshrq_n_s32(vreinterpretq_s32_f32(x), 23), vdupq_n_s32(0x7E)));
  x = vreinterpretq_f32_s32(
      vorrq_s32(vandq_s32(vreinterpretq_s32_f32(x), vdupq_n_s32(static_cast<int>(0x807FFFFFU))),
                vreinterpretq_s32_f32(vdupq_n_f32(0.5F))));
  const uint32x4_t lo = vcltq_f32(x, vdupq_n_f32(sqrt_half));
  e = vsubq_f32(e, vbslq_f32(lo, vdupq_n_f32(1.0F), vdupq_n_f32(0.0F)));
  x = vsubq_f32(vaddq_f32(x, vbslq_f32(lo, x, vdupq_n_f32(0.0F))), vdupq_n_f32(1.0F));
  const float32x4_t z = vmulq_f32(x, x);
  float32x4_t y = vdupq_n_f32(log_p0);
  y = vfmaq_f32(vdupq_n_f32(log_p1), y, x);
  y = vfmaq_f32(vdupq_n_f32(log_p2), y, x);
  y = vfmaq_f32(vdupq_n_f32(log_p3), y, x);
  y = vfmaq_f32(vdupq_n_f32(log_p4), y, x);
  y = vfmaq_f32(vdupq_n_f32(log_p5), y, x);
  y = vfmaq_f32(vdupq_n_f32(log_p6), y, x);
  y = vfmaq_f32(vdupq_n_f32(log_p7), y, x);
  y = vfmaq_f32(vdupq_n_f32(log_p8), y, x);
  y = vmulq_f32(vmulq_f32(y, x), z);
  y = vfmaq_f32(y, e, vdupq_n_f32(ln2_lo));
  y = vfmsq_f32(y, z, vdupq_n_f32(0.5F));
  x = vaddq_f32(x, y);
  return vfmaq_f32(x, e, vdupq_n_f32(ln2_hi));
}
#endif

#if defined(__AVX2__)
float hsum8(__m256 v) noexcept
{
  const __m128 lo = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  const __m128 s2 = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(_mm_hadd_ps(s2, s2));
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

namespace {

// 1 recurrence step, state-major [n][c]: advance h in place, emit y (skip term + Σ_n h·c)
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
  std::size_t i{0uz};
#if defined(__AVX2__)
  const __m256 one = _mm256_set1_ps(1.0F);
  const __m256 zero = _mm256_setzero_ps();
  for (; i + 8uz <= x.size(); i += 8uz) {
    const __m256 v = _mm256_loadu_ps(x.data() + i);
    _mm256_storeu_ps(x.data() + i,
                     _mm256_div_ps(v, _mm256_add_ps(one, exp8(_mm256_sub_ps(zero, v)))));
  }
#elif defined(__ARM_NEON)
  const float32x4_t one = vdupq_n_f32(1.0F);
  for (; i + 4uz <= x.size(); i += 4uz) {
    const float32x4_t v = vld1q_f32(x.data() + i);
    vst1q_f32(x.data() + i, vdivq_f32(v, vaddq_f32(one, exp4(vnegq_f32(v)))));
  }
#endif
  for (; i < x.size(); ++i)
    x[i] = x[i] / (1.0F + std::exp(-x[i]));
}

void softplus(std::span<float> x) noexcept
{
  // max(v,0) + log(1 + e^-|v|)
  // e^-|v| in (0,1]
  std::size_t i{0uz};
#if defined(__AVX2__)
  const __m256 one = _mm256_set1_ps(1.0F);
  const __m256 zero = _mm256_setzero_ps();
  const __m256 sign = _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int>(0x80000000U)));
  for (; i + 8uz <= x.size(); i += 8uz) {
    const __m256 v = _mm256_loadu_ps(x.data() + i);
    const __m256 nabs = _mm256_or_ps(v, sign); // -|v|
    _mm256_storeu_ps(x.data() + i,
                     _mm256_add_ps(_mm256_max_ps(v, zero), log8(_mm256_add_ps(one, exp8(nabs)))));
  }
#elif defined(__ARM_NEON)
  const float32x4_t one = vdupq_n_f32(1.0F);
  const float32x4_t zero = vdupq_n_f32(0.0F);
  for (; i + 4uz <= x.size(); i += 4uz) {
    const float32x4_t v = vld1q_f32(x.data() + i);
    const float32x4_t nabs = vnegq_f32(vabsq_f32(v));
    vst1q_f32(x.data() + i, vaddq_f32(vmaxq_f32(v, zero), log4(vaddq_f32(one, exp4(nabs)))));
  }
#endif
  for (; i < x.size(); ++i)
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
  if (rows == 1uz) { // single vector
    const float* __restrict__ ir = in.data();
    for (std::size_t o{0uz}; o < out_dim; ++o) {
      const float* __restrict__ wr = w.data() + (o * in_dim);
      float acc{0.0F};
      std::size_t i{0uz};
#if defined(__AVX2__)
      __m256 a0 = _mm256_setzero_ps();
      __m256 a1 = a0;
      __m256 a2 = a0;
      __m256 a3 = a0;
      for (; i + 32uz <= in_dim; i += 32uz) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(ir + i), _mm256_loadu_ps(wr + i), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(ir + i + 8uz), _mm256_loadu_ps(wr + i + 8uz), a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(ir + i + 16uz), _mm256_loadu_ps(wr + i + 16uz), a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(ir + i + 24uz), _mm256_loadu_ps(wr + i + 24uz), a3);
      }
      __m256 av = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
      for (; i + 8uz <= in_dim; i += 8uz)
        av = _mm256_fmadd_ps(_mm256_loadu_ps(ir + i), _mm256_loadu_ps(wr + i), av);
      acc = hsum8(av);
#elif defined(__ARM_NEON)
      float32x4_t a0 = vdupq_n_f32(0.0F);
      float32x4_t a1 = a0;
      float32x4_t a2 = a0;
      float32x4_t a3 = a0;
      for (; i + 16uz <= in_dim; i += 16uz) {
        a0 = vfmaq_f32(a0, vld1q_f32(ir + i), vld1q_f32(wr + i));
        a1 = vfmaq_f32(a1, vld1q_f32(ir + i + 4uz), vld1q_f32(wr + i + 4uz));
        a2 = vfmaq_f32(a2, vld1q_f32(ir + i + 8uz), vld1q_f32(wr + i + 8uz));
        a3 = vfmaq_f32(a3, vld1q_f32(ir + i + 12uz), vld1q_f32(wr + i + 12uz));
      }
      float32x4_t av = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
      for (; i + 4uz <= in_dim; i += 4uz)
        av = vfmaq_f32(av, vld1q_f32(ir + i), vld1q_f32(wr + i));
      acc = vaddvq_f32(av);
#endif
      for (; i < in_dim; ++i)
        acc += ir[i] * wr[i];
      out.data()[o] = acc;
    }
    return;
  }

  for (std::size_t o{0uz}; o < out_dim; ++o) {
    const float* __restrict__ wr = w.data() + (o * in_dim);
    for (std::size_t r0{0uz}; r0 < rows; r0 += 4uz) {
      const std::size_t nr = (rows - r0 < 4uz) ? (rows - r0) : 4uz; // tail rows clamp to r0
      const float* __restrict__ i0 = in.data() + ((r0 + 0uz) * in_dim);
      const float* __restrict__ i1 = in.data() + ((r0 + (nr > 1uz ? 1uz : 0uz)) * in_dim);
      const float* __restrict__ i2 = in.data() + ((r0 + (nr > 2uz ? 2uz : 0uz)) * in_dim);
      const float* __restrict__ i3 = in.data() + ((r0 + (nr > 3uz ? 3uz : 0uz)) * in_dim);
      float acc0{0.0F};
      float acc1{0.0F};
      float acc2{0.0F};
      float acc3{0.0F};
      std::size_t i{0uz};
#if defined(__AVX2__)
      __m256 v0 = _mm256_setzero_ps();
      __m256 v1 = v0;
      __m256 v2 = v0;
      __m256 v3 = v0;
      for (; i + 8uz <= in_dim; i += 8uz) {
        const __m256 wv = _mm256_loadu_ps(wr + i);
        v0 = _mm256_fmadd_ps(_mm256_loadu_ps(i0 + i), wv, v0);
        v1 = _mm256_fmadd_ps(_mm256_loadu_ps(i1 + i), wv, v1);
        v2 = _mm256_fmadd_ps(_mm256_loadu_ps(i2 + i), wv, v2);
        v3 = _mm256_fmadd_ps(_mm256_loadu_ps(i3 + i), wv, v3);
      }
      acc0 = hsum8(v0);
      acc1 = hsum8(v1);
      acc2 = hsum8(v2);
      acc3 = hsum8(v3);
#elif defined(__ARM_NEON)
      float32x4_t v0 = vdupq_n_f32(0.0F);
      float32x4_t v1 = v0;
      float32x4_t v2 = v0;
      float32x4_t v3 = v0;
      for (; i + 4uz <= in_dim; i += 4uz) {
        const float32x4_t wv = vld1q_f32(wr + i);
        v0 = vfmaq_f32(v0, vld1q_f32(i0 + i), wv);
        v1 = vfmaq_f32(v1, vld1q_f32(i1 + i), wv);
        v2 = vfmaq_f32(v2, vld1q_f32(i2 + i), wv);
        v3 = vfmaq_f32(v3, vld1q_f32(i3 + i), wv);
      }
      acc0 = vaddvq_f32(v0);
      acc1 = vaddvq_f32(v1);
      acc2 = vaddvq_f32(v2);
      acc3 = vaddvq_f32(v3);
#endif
      for (; i < in_dim; ++i) {
        const float wv = wr[i];
        acc0 += i0[i] * wv;
        acc1 += i1[i] * wv;
        acc2 += i2[i] * wv;
        acc3 += i3[i] * wv;
      }
      float* __restrict__ orow = out.data() + (r0 * out_dim);
      orow[o] = acc0;
      if (nr > 1uz)
        orow[out_dim + o] = acc1;
      if (nr > 2uz)
        orow[(2uz * out_dim) + o] = acc2;
      if (nr > 3uz)
        orow[(3uz * out_dim) + o] = acc3;
    }
  }
}

void discretize(std::span<const float> delta, std::span<const float> a_log,
                std::span<const float> b, std::span<const float> u, std::span<float> delta_a,
                std::span<float> delta_bu, std::span<float> a_work, std::size_t length,
                std::size_t d_inner, std::size_t d_state) noexcept
{
  const std::size_t plane = d_inner * d_state;
  // A = -exp(a_log): vectorized exp into delta_a
  // transpose + negate into a_work[n][c].
  float* __restrict__ tmp = delta_a.data();
  std::size_t i{0uz};
#if defined(__AVX2__)
  for (; i + 8uz <= plane; i += 8uz)
    _mm256_storeu_ps(tmp + i, exp8(_mm256_loadu_ps(a_log.data() + i)));
#elif defined(__ARM_NEON)
  for (; i + 4uz <= plane; i += 4uz)
    vst1q_f32(tmp + i, exp4(vld1q_f32(a_log.data() + i)));
#endif
  for (; i < plane; ++i)
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
      std::size_t c{0uz};
#if defined(__AVX2__)
      const __m256 vbn = _mm256_set1_ps(bn);
      for (; c + 8uz <= d_inner; c += 8uz) {
        const __m256 vdt = _mm256_loadu_ps(dt + c);
        _mm256_storeu_ps(da_n + c, exp8(_mm256_mul_ps(vdt, _mm256_loadu_ps(an + c))));
        _mm256_storeu_ps(dbu_n + c,
                         _mm256_mul_ps(_mm256_mul_ps(vdt, vbn), _mm256_loadu_ps(ut + c)));
      }
#elif defined(__ARM_NEON)
      const float32x4_t vbn = vdupq_n_f32(bn);
      for (; c + 4uz <= d_inner; c += 4uz) {
        const float32x4_t vdt = vld1q_f32(dt + c);
        vst1q_f32(da_n + c, exp4(vmulq_f32(vdt, vld1q_f32(an + c))));
        vst1q_f32(dbu_n + c, vmulq_f32(vmulq_f32(vdt, vbn), vld1q_f32(ut + c)));
      }
#endif
      for (; c < d_inner; ++c) {
        da_n[c] = std::exp(dt[c] * an[c]);
        dbu_n[c] = dt[c] * bn * ut[c];
      }
    }
  }
}

} // namespace fe