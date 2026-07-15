#include "kernels.h"

#include <cmath>
#include <cstddef>
#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#define FE_RESTRICT __restrict__

namespace fe {
namespace {

#if defined(__AVX2__)
// Cephes 8-wide expf ~1 ULP, LLVM folds an equivalent scalar
// polynomial back into a libm exp()
__m256 exp8(__m256 x) noexcept
{
  x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-88.3762F)), _mm256_set1_ps(88.3762F));
  const __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504F)),
                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  // Cephes two-part ln2 hi+lo for reduction
  x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375F), x); // NOLINT(modernize-use-std-numbers)
  x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4F), x);
  __m256 y = _mm256_set1_ps(1.9875691500e-4F);
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3F));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3F));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2F));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1F));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1F));
  y = _mm256_fmadd_ps(y, _mm256_mul_ps(x, x), x);
  y = _mm256_add_ps(y, _mm256_set1_ps(1.0F));
  const __m256i pow2 =
      _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(fx), _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(y, _mm256_castsi256_ps(pow2));
}
#elif defined(__ARM_NEON)
// Same Cephes expf, 4-wide NEON.
float32x4_t exp4(float32x4_t x) noexcept
{
  x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-88.3762F)), vdupq_n_f32(88.3762F));
  const float32x4_t fx = vrndnq_f32(vmulq_n_f32(x, 1.44269504F));
  x = vfmsq_f32(x, fx, vdupq_n_f32(0.693359375F)); // a - b·c
  x = vfmsq_f32(x, fx, vdupq_n_f32(-2.12194440e-4F));
  float32x4_t y = vdupq_n_f32(1.9875691500e-4F);
  y = vfmaq_f32(vdupq_n_f32(1.3981999507e-3F), y, x); // c + a·b
  y = vfmaq_f32(vdupq_n_f32(8.3334519073e-3F), y, x);
  y = vfmaq_f32(vdupq_n_f32(4.1665795894e-2F), y, x);
  y = vfmaq_f32(vdupq_n_f32(1.6666665459e-1F), y, x);
  y = vfmaq_f32(vdupq_n_f32(5.0000001201e-1F), y, x);
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
    const float* FE_RESTRICT xc = x.data() + (c * length);
    const float* FE_RESTRICT wc = weight.data() + (c * kernel);
    float* FE_RESTRICT yc = y.data() + (c * length);

    const float bc = bias[c];
    for (std::size_t t{0uz}; t < length; ++t)
      yc[t] = bc;

    // One tap at a time as a SAXPY over time — branchless, auto-vectorizes.
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
  float* FE_RESTRICT hs = h.data();
  for (std::size_t i{0uz}; i < d_inner * d_state; ++i)
    hs[i] = 0.0F; // h_0 = 0

  const std::size_t plane = d_state * d_inner;
  const float* FE_RESTRICT d = d_skip.data();
  for (std::size_t t{0uz}; t < length; ++t) {
    const float* FE_RESTRICT da_t = delta_a.data() + (t * plane);
    const float* FE_RESTRICT dbu_t = delta_bu.data() + (t * plane);
    const float* FE_RESTRICT c_t = c_proj.data() + (t * d_state);
    const float* FE_RESTRICT u_t = u.data() + (t * d_inner);
    float* FE_RESTRICT y_t = y.data() + (t * d_inner);

    for (std::size_t c{0uz}; c < d_inner; ++c)
      y_t[c] = d[c] * u_t[c]; // skip term, scan accumulates onto it

    // State-major [n][c]
    // 4x interleave
    for (std::size_t n{0uz}; n < d_state; ++n) {
      float* FE_RESTRICT hn = hs + (n * d_inner);
      const float* FE_RESTRICT da_n = da_t + (n * d_inner);
      const float* FE_RESTRICT dbu_n = dbu_t + (n * d_inner);
      const float cn = c_t[n];
      for (std::size_t c{0uz}; c < d_inner; ++c) {
        hn[c] = (da_n[c] * hn[c]) + dbu_n[c];
        y_t[c] += hn[c] * cn;
      }
    }
  }
}

} // namespace fe
