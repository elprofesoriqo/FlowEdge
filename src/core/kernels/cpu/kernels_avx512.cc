#include "arena/thread_pool.h"
#include "kernels/cpu/cephes.h"
#include "kernels/kernels.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace fe {
namespace {

// Cephes 8-wide expf (~1 ULP)
__m512 exp16(__m512 x) noexcept
{
  x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-exp_clamp)), _mm512_set1_ps(exp_clamp));
  const __m512 fx = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(log2e)),
                                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  x = _mm512_fnmadd_ps(fx, _mm512_set1_ps(ln2_hi), x);
  x = _mm512_fnmadd_ps(fx, _mm512_set1_ps(ln2_lo), x);
  __m512 y = _mm512_set1_ps(exp_p0);
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(exp_p1));
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(exp_p2));
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(exp_p3));
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(exp_p4));
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(exp_p5));
  y = _mm512_fmadd_ps(y, _mm512_mul_ps(x, x), x);
  y = _mm512_add_ps(y, _mm512_set1_ps(1.0F));
  const __m512i pow2 =
      _mm512_slli_epi32(_mm512_add_epi32(_mm512_cvttps_epi32(fx), _mm512_set1_epi32(127)), 23);
  return _mm512_mul_ps(y, _mm512_castsi512_ps(pow2));
}

// Cephes 8-wide logf (~1 ULP)
__m512 log16(__m512 x) noexcept
{
  __m512 e = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_srli_epi32(_mm512_castps_si512(x), 23),
                                                 _mm512_set1_epi32(0x7E))); // unbiased exp
  x = _mm512_or_ps(_mm512_and_ps(x, _mm512_castsi512_ps(
                                        _mm512_set1_epi32(static_cast<int>(0x807FFFFFU)))),
                   _mm512_set1_ps(0.5F)); // mantissa in [0.5,1)
  __mmask16 m = _mm512_cmp_ps_mask(x, _mm512_set1_ps(sqrt_half), _CMP_LT_OS);
  const __m512 lo = _mm512_castsi512_ps(_mm512_maskz_set1_epi32(m, 0xFFFFFFFF));
  e = _mm512_sub_ps(e, _mm512_and_ps(_mm512_set1_ps(1.0F), lo));
  x = _mm512_sub_ps(_mm512_add_ps(x, _mm512_and_ps(x, lo)), _mm512_set1_ps(1.0F));
  const __m512 z = _mm512_mul_ps(x, x);
  __m512 y = _mm512_set1_ps(log_p0);
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(log_p1));
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(log_p2));
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(log_p3));
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(log_p4));
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(log_p5));
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(log_p6));
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(log_p7));
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(log_p8));
  y = _mm512_mul_ps(_mm512_mul_ps(y, x), z);
  y = _mm512_fmadd_ps(e, _mm512_set1_ps(ln2_lo), y);
  y = _mm512_fnmadd_ps(z, _mm512_set1_ps(0.5F), y);
  x = _mm512_add_ps(x, y);
  return _mm512_fmadd_ps(e, _mm512_set1_ps(ln2_hi), x);
}

float hsum16(__m512 v) noexcept
{
  __m256 lo256 = _mm256_add_ps(_mm512_castps512_ps256(v), _mm512_extractf32x8_ps(v, 1));
  __m128 lo128 = _mm_add_ps(_mm256_castps256_ps128(lo256), _mm256_extractf128_ps(lo256, 1));
  __m128 s2 = _mm_hadd_ps(lo128, lo128);
  return _mm_cvtss_f32(_mm_hadd_ps(s2, s2));
}

// widen 16 BF16 → 16 F32
__m512 load_bf16_16(const uint16_t* p) noexcept
{
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256(
                                                   reinterpret_cast<const __m256i*>(p))),
                                               16));
}

// scalar BF16 → F32
inline float bf16_to_f32(uint16_t v) noexcept
{
  const uint32_t bits = static_cast<uint32_t>(v) << 16;
  return std::bit_cast<float>(bits);
}

} // namespace

void gate_silu(std::span<const float> a, std::span<const float> g, std::span<float> out) noexcept
{
  std::size_t i{0uz};
  const __m512 one = _mm512_set1_ps(1.0F);
  const __m512 zero = _mm512_setzero_ps();
  for (; i + 16uz <= out.size(); i += 16uz) {
    const __m512 vg = _mm512_loadu_ps(g.data() + i);
    const __m512 sig = _mm512_div_ps(vg, _mm512_add_ps(one, exp16(_mm512_sub_ps(zero, vg))));
    _mm512_storeu_ps(out.data() + i, _mm512_mul_ps(_mm512_loadu_ps(a.data() + i), sig));
  }
  for (; i < out.size(); ++i) {
    const float v = g[i];
    out[i] = a[i] * (v / (1.0F + std::exp(-v)));
  }
}

void silu(std::span<float> x) noexcept
{
  std::size_t i{0uz};
  const __m512 one = _mm512_set1_ps(1.0F);
  const __m512 zero = _mm512_setzero_ps();
  for (; i + 16uz <= x.size(); i += 16uz) {
    const __m512 v = _mm512_loadu_ps(x.data() + i);
    _mm512_storeu_ps(x.data() + i,
                     _mm512_div_ps(v, _mm512_add_ps(one, exp16(_mm512_sub_ps(zero, v)))));
  }
  for (; i < x.size(); ++i)
    x[i] = x[i] / (1.0F + std::exp(-x[i]));
}

void softplus(std::span<float> x) noexcept
{
  // max(v,0) + log(1 + e^-|v|)
  // e^-|v| in (0,1]
  std::size_t i{0uz};
  const __m512 one = _mm512_set1_ps(1.0F);
  const __m512 zero = _mm512_setzero_ps();
  const __m512 sign = _mm512_castsi512_ps(_mm512_set1_epi32(static_cast<int>(0x80000000U)));
  for (; i + 16uz <= x.size(); i += 16uz) {
    const __m512 v = _mm512_loadu_ps(x.data() + i);
    const __m512 nabs = _mm512_or_ps(v, sign); // -|v|
    _mm512_storeu_ps(x.data() + i,
                     _mm512_add_ps(_mm512_max_ps(v, zero), log16(_mm512_add_ps(one, exp16(nabs)))));
  }
  for (; i < x.size(); ++i)
    x[i] = std::log1p(std::exp(x[i]));
}

namespace {
struct MatmulF32Row1Ctx
{
  std::span<const float> in;
  std::span<const float> w;
  std::span<float> out;
  std::size_t in_dim;
};
void matmul_f32_row1(std::size_t lo, std::size_t hi, const MatmulF32Row1Ctx* ctx) noexcept
{
  const float* __restrict__ ir = ctx->in.data();
  const float* __restrict__ wd = ctx->w.data();
  float* __restrict__ od = ctx->out.data();
  const std::size_t in_dim = ctx->in_dim;
  for (std::size_t o{lo}; o < hi; ++o) {
    const float* __restrict__ wr = wd + (o * in_dim);
    float acc{0.0F};
    std::size_t i{0uz};
    __m512 a0 = _mm512_setzero_ps();
    __m512 a1 = a0;
    __m512 a2 = a0;
    __m512 a3 = a0;
    for (; i + 64uz <= in_dim; i += 64uz) {
      a0 = _mm512_fmadd_ps(_mm512_loadu_ps(ir + i), _mm512_loadu_ps(wr + i), a0);
      a1 = _mm512_fmadd_ps(_mm512_loadu_ps(ir + i + 16uz), _mm512_loadu_ps(wr + i + 16uz), a1);
      a2 = _mm512_fmadd_ps(_mm512_loadu_ps(ir + i + 32uz), _mm512_loadu_ps(wr + i + 32uz), a2);
      a3 = _mm512_fmadd_ps(_mm512_loadu_ps(ir + i + 48uz), _mm512_loadu_ps(wr + i + 48uz), a3);
    }
    __m512 av = _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3));
    for (; i + 16uz <= in_dim; i += 16uz)
      av = _mm512_fmadd_ps(_mm512_loadu_ps(ir + i), _mm512_loadu_ps(wr + i), av);
    acc = hsum16(av);
    for (; i < in_dim; ++i)
      acc += ir[i] * wr[i];
    od[o] = acc;
  }
}
struct MatmulF32Row1Op
{
  const MatmulF32Row1Ctx* ctx;
  void operator()(std::size_t lo, std::size_t hi) const noexcept { matmul_f32_row1(lo, hi, ctx); }
};

struct MatmulF32RowNCtx
{
  std::span<const float> in;
  std::span<const float> w;
  std::span<float> out;
  std::size_t rows;
  std::size_t in_dim;
  std::size_t out_dim;
};
void matmul_f32_rown(std::size_t lo, std::size_t hi, const MatmulF32RowNCtx* ctx) noexcept
{
  const float* __restrict__ in_data = ctx->in.data();
  const float* __restrict__ w_data = ctx->w.data();
  float* __restrict__ out_data = ctx->out.data();
  const std::size_t rows = ctx->rows;
  const std::size_t in_dim = ctx->in_dim;
  const std::size_t out_dim = ctx->out_dim;
  for (std::size_t o{lo}; o < hi; ++o) {
    const float* __restrict__ wr = w_data + (o * in_dim);
    for (std::size_t r0{0uz}; r0 < rows; r0 += 4uz) {
      const std::size_t nr = (rows - r0 < 4uz) ? (rows - r0) : 4uz;
      const float* __restrict__ i0 = in_data + ((r0 + 0uz) * in_dim);
      const float* __restrict__ i1 = in_data + ((r0 + (nr > 1uz ? 1uz : 0uz)) * in_dim);
      const float* __restrict__ i2 = in_data + ((r0 + (nr > 2uz ? 2uz : 0uz)) * in_dim);
      const float* __restrict__ i3 = in_data + ((r0 + (nr > 3uz ? 3uz : 0uz)) * in_dim);
      float acc0{0.0F}, acc1{0.0F}, acc2{0.0F}, acc3{0.0F};
      std::size_t i{0uz};
      __m512 v0 = _mm512_setzero_ps();
      __m512 v1 = v0, v2 = v0, v3 = v0;
      for (; i + 16uz <= in_dim; i += 16uz) {
        const __m512 wv = _mm512_loadu_ps(wr + i);
        v0 = _mm512_fmadd_ps(_mm512_loadu_ps(i0 + i), wv, v0);
        if (nr > 1)
          v1 = _mm512_fmadd_ps(_mm512_loadu_ps(i1 + i), wv, v1);
        if (nr > 2)
          v2 = _mm512_fmadd_ps(_mm512_loadu_ps(i2 + i), wv, v2);
        if (nr > 3)
          v3 = _mm512_fmadd_ps(_mm512_loadu_ps(i3 + i), wv, v3);
      }
      acc0 = hsum16(v0);
      if (nr > 1)
        acc1 = hsum16(v1);
      if (nr > 2)
        acc2 = hsum16(v2);
      if (nr > 3)
        acc3 = hsum16(v3);
      for (; i < in_dim; ++i) {
        const float wv = wr[i];
        acc0 += i0[i] * wv;
        if (nr > 1)
          acc1 += i1[i] * wv;
        if (nr > 2)
          acc2 += i2[i] * wv;
        if (nr > 3)
          acc3 += i3[i] * wv;
      }
      float* __restrict__ orow = out_data + (r0 * out_dim);
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
struct MatmulF32RowNOp
{
  const MatmulF32RowNCtx* ctx;
  void operator()(std::size_t lo, std::size_t hi) const noexcept { matmul_f32_rown(lo, hi, ctx); }
};

struct MatmulU16Row1Ctx
{
  std::span<const float> in;
  std::span<const uint16_t> w;
  std::span<float> out;
  std::size_t in_dim;
};
void matmul_u16_row1(std::size_t lo, std::size_t hi, const MatmulU16Row1Ctx* ctx) noexcept
{
  const float* __restrict__ ir = ctx->in.data();
  const uint16_t* __restrict__ wd = ctx->w.data();
  float* __restrict__ od = ctx->out.data();
  const std::size_t in_dim = ctx->in_dim;
  for (std::size_t o{lo}; o < hi; ++o) {
    const uint16_t* __restrict__ wr = wd + (o * in_dim);
    float acc{0.0F};
    std::size_t i{0uz};
    __m512 a0 = _mm512_setzero_ps();
    __m512 a1 = a0;
    __m512 a2 = a0;
    __m512 a3 = a0;
    for (; i + 128uz <= in_dim; i += 128uz) {
      __m512bh i_bh0 =
          (__m512bh)_mm512_inserti32x8(_mm512_castsi256_si512(
                                           (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(ir + i))),
                                       (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(ir + i + 16uz)),
                                       1);
      __m512bh w_bh0 = (__m512bh)_mm512_loadu_si512(reinterpret_cast<const void*>(wr + i));
      a0 = _mm512_dpbf16_ps(a0, i_bh0, w_bh0);

      __m512bh i_bh1 =
          (__m512bh)_mm512_inserti32x8(_mm512_castsi256_si512((__m256i)_mm512_cvtneps_pbh(
                                           _mm512_loadu_ps(ir + i + 32uz))),
                                       (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(ir + i + 48uz)),
                                       1);
      __m512bh w_bh1 = (__m512bh)_mm512_loadu_si512(reinterpret_cast<const void*>(wr + i + 32uz));
      a1 = _mm512_dpbf16_ps(a1, i_bh1, w_bh1);

      __m512bh i_bh2 =
          (__m512bh)_mm512_inserti32x8(_mm512_castsi256_si512((__m256i)_mm512_cvtneps_pbh(
                                           _mm512_loadu_ps(ir + i + 64uz))),
                                       (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(ir + i + 80uz)),
                                       1);
      __m512bh w_bh2 = (__m512bh)_mm512_loadu_si512(reinterpret_cast<const void*>(wr + i + 64uz));
      a2 = _mm512_dpbf16_ps(a2, i_bh2, w_bh2);

      __m512bh i_bh3 =
          (__m512bh)_mm512_inserti32x8(_mm512_castsi256_si512((__m256i)_mm512_cvtneps_pbh(
                                           _mm512_loadu_ps(ir + i + 96uz))),
                                       (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(ir + i + 112uz)),
                                       1);
      __m512bh w_bh3 = (__m512bh)_mm512_loadu_si512(reinterpret_cast<const void*>(wr + i + 96uz));
      a3 = _mm512_dpbf16_ps(a3, i_bh3, w_bh3);
    }
    __m512 av = _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3));
    for (; i + 16uz <= in_dim; i += 16uz)
      av = _mm512_fmadd_ps(_mm512_loadu_ps(ir + i), load_bf16_16(wr + i), av);
    acc = hsum16(av);
    for (; i < in_dim; ++i)
      acc += ir[i] * bf16_to_f32(wr[i]);
    od[o] = acc;
  }
}
struct MatmulU16Row1Op
{
  const MatmulU16Row1Ctx* ctx;
  void operator()(std::size_t lo, std::size_t hi) const noexcept { matmul_u16_row1(lo, hi, ctx); }
};

struct MatmulU16RowNCtx
{
  std::span<const float> in;
  std::span<const uint16_t> w;
  std::span<float> out;
  std::size_t rows;
  std::size_t in_dim;
  std::size_t out_dim;
};
void matmul_u16_rown(std::size_t lo, std::size_t hi, const MatmulU16RowNCtx* ctx) noexcept
{
  const float* __restrict__ in_data = ctx->in.data();
  const uint16_t* __restrict__ w_data = ctx->w.data();
  float* __restrict__ out_data = ctx->out.data();
  const std::size_t rows = ctx->rows;
  const std::size_t in_dim = ctx->in_dim;
  const std::size_t out_dim = ctx->out_dim;
  for (std::size_t o{lo}; o < hi; ++o) {
    const uint16_t* __restrict__ wr = w_data + (o * in_dim);
    for (std::size_t r0{0uz}; r0 < rows; r0 += 4uz) {
      const std::size_t nr = (rows - r0 < 4uz) ? (rows - r0) : 4uz;
      const float* __restrict__ i0 = in_data + ((r0 + 0uz) * in_dim);
      const float* __restrict__ i1 = in_data + ((r0 + (nr > 1uz ? 1uz : 0uz)) * in_dim);
      const float* __restrict__ i2 = in_data + ((r0 + (nr > 2uz ? 2uz : 0uz)) * in_dim);
      const float* __restrict__ i3 = in_data + ((r0 + (nr > 3uz ? 3uz : 0uz)) * in_dim);
      float acc0{0.0F}, acc1{0.0F}, acc2{0.0F}, acc3{0.0F};
      std::size_t i{0uz};
      __m512 v0 = _mm512_setzero_ps();
      __m512 v1 = v0, v2 = v0, v3 = v0;
      for (; i + 32uz <= in_dim; i += 32uz) {
        __m512bh w_bh = (__m512bh)_mm512_loadu_si512(reinterpret_cast<const void*>(wr + i));

        __m512bh i_bh0 = (__m512bh)_mm512_inserti32x8(
            _mm512_castsi256_si512((__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(i0 + i))),
            (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(i0 + i + 16uz)), 1);
        v0 = _mm512_dpbf16_ps(v0, i_bh0, w_bh);
        if (nr > 1) {
          __m512bh i_bh1 = (__m512bh)_mm512_inserti32x8(
              _mm512_castsi256_si512((__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(i1 + i))),
              (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(i1 + i + 16uz)), 1);
          v1 = _mm512_dpbf16_ps(v1, i_bh1, w_bh);
        }
        if (nr > 2) {
          __m512bh i_bh2 = (__m512bh)_mm512_inserti32x8(
              _mm512_castsi256_si512((__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(i2 + i))),
              (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(i2 + i + 16uz)), 1);
          v2 = _mm512_dpbf16_ps(v2, i_bh2, w_bh);
        }
        if (nr > 3) {
          __m512bh i_bh3 = (__m512bh)_mm512_inserti32x8(
              _mm512_castsi256_si512((__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(i3 + i))),
              (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(i3 + i + 16uz)), 1);
          v3 = _mm512_dpbf16_ps(v3, i_bh3, w_bh);
        }
      }
      acc0 = hsum16(v0);
      if (nr > 1)
        acc1 = hsum16(v1);
      if (nr > 2)
        acc2 = hsum16(v2);
      if (nr > 3)
        acc3 = hsum16(v3);
      for (; i < in_dim; ++i) {
        const float wf = bf16_to_f32(wr[i]);
        acc0 += i0[i] * wf;
        if (nr > 1)
          acc1 += i1[i] * wf;
        if (nr > 2)
          acc2 += i2[i] * wf;
        if (nr > 3)
          acc3 += i3[i] * wf;
      }
      float* __restrict__ orow = out_data + (r0 * out_dim);
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
struct MatmulU16RowNOp
{
  const MatmulU16RowNCtx* ctx;
  void operator()(std::size_t lo, std::size_t hi) const noexcept { matmul_u16_rown(lo, hi, ctx); }
};
} // namespace

void matmul(std::span<const float> in, std::span<const float> w, std::span<float> out,
            std::size_t rows, std::size_t in_dim, std::size_t out_dim, ThreadPool* pool) noexcept
{
  const unsigned n = (pool != nullptr) ? pool->nthreads() : 0u;
  if (rows == 1uz) { // single vector
    MatmulF32Row1Ctx ctx{in, w, out, in_dim};
    MatmulF32Row1Op op{&ctx};
    if (pool != nullptr && n > 0u && (out_dim / n) >= 64uz)
      parallel_for(*pool, out_dim, op);
    else
      op(0uz, out_dim);
    return;
  }

  MatmulF32RowNCtx ctx{in, w, out, rows, in_dim, out_dim};
  MatmulF32RowNOp op{&ctx};
  if (pool != nullptr && rows > 1uz && n > 0u && (out_dim / n) >= 64uz)
    parallel_for(*pool, out_dim, op);
  else
    op(0uz, out_dim);
}

void matmul(std::span<const float> in, std::span<const uint16_t> w, std::span<float> out,
            std::size_t rows, std::size_t in_dim, std::size_t out_dim, ThreadPool* pool) noexcept
{
  const unsigned n = (pool != nullptr) ? pool->nthreads() : 0u;
  if (rows == 1uz) { // single vector
    MatmulU16Row1Ctx ctx{in, w, out, in_dim};
    MatmulU16Row1Op op{&ctx};
    if (pool != nullptr && n > 0u && (out_dim / n) >= 64uz)
      parallel_for(*pool, out_dim, op);
    else
      op(0uz, out_dim);
    return;
  }

  MatmulU16RowNCtx ctx{in, w, out, rows, in_dim, out_dim};
  MatmulU16RowNOp op{&ctx};
  if (pool != nullptr && rows > 1uz && n > 0u && (out_dim / n) >= 64uz)
    parallel_for(*pool, out_dim, op);
  else
    op(0uz, out_dim);
}

void conv1d_causal(std::span<const float> x, std::span<const float> weight,
                   std::span<const float> bias, std::span<float> y, std::size_t channels,
                   std::size_t length, std::size_t kernel) noexcept
{
  for (std::size_t c{0uz}; c < channels; ++c) {
    const float* __restrict__ xc = x.data() + (c * length);
    const float* __restrict__ wc = weight.data() + (c * kernel);
    float* __restrict__ yc = y.data() + (c * length);
#if defined(__GNUC__) && !defined(__clang__)
    if (length < 16uz) {
      const float bc = bias[c];
      for (std::size_t t{0uz}; t < length; ++t)
        yc[t] = bc;
      for (std::size_t k{0uz}; k < kernel; ++k) {
        const float wk = wc[k];
        const std::size_t start = (kernel - 1uz) - k;
        for (std::size_t t{start}; t < length; ++t)
          yc[t] += wk * xc[t - start];
      }
      continue;
    }
#endif
    const __m512 vbc = _mm512_set1_ps(bias[c]);
    std::size_t t{0uz};
    for (; t + 16uz <= length; t += 16uz)
      _mm512_storeu_ps(yc + t, vbc);
    for (; t < length; ++t)
      yc[t] = bias[c];

    for (std::size_t k{0uz}; k < kernel; ++k) {
      const __m512 vwk = _mm512_set1_ps(wc[k]);
      const std::size_t start = (kernel - 1uz) - k;
      std::size_t ts = start;
      for (; ts + 16uz <= length; ts += 16uz) {
        _mm512_storeu_ps(yc + ts, _mm512_fmadd_ps(vwk, _mm512_loadu_ps(xc + ts - start),
                                                  _mm512_loadu_ps(yc + ts)));
      }
      for (; ts < length; ++ts)
        yc[ts] += wc[k] * xc[ts - start];
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
    __m512 vss = _mm512_setzero_ps();
    std::size_t i{0uz};
    for (; i + 16uz <= dim; i += 16uz) {
      const __m512 v = _mm512_loadu_ps(ir + i);
      vss = _mm512_fmadd_ps(v, v, vss);
    }
    float ss = hsum16(vss);
    for (; i < dim; ++i)
      ss += ir[i] * ir[i];
    const float scale = 1.0F / std::sqrt((ss / static_cast<float>(dim)) + eps);
    const __m512 vscale = _mm512_set1_ps(scale);
    i = 0uz;
    for (; i + 16uz <= dim; i += 16uz)
      _mm512_storeu_ps(orow + i,
                       _mm512_mul_ps(vscale, _mm512_mul_ps(_mm512_loadu_ps(ir + i),
                                                           _mm512_loadu_ps(weight.data() + i))));
    for (; i < dim; ++i)
      orow[i] = ir[i] * scale * weight[i];
  }
}

void discretize_and_scan(std::span<const float> delta, std::span<const float> a_neg,
                         std::span<const float> b, std::span<const float> u,
                         std::span<const float> c_proj, std::span<const float> d_skip,
                         std::span<float> h, std::span<float> y, std::size_t length,
                         std::size_t d_inner, std::size_t d_state, bool reset_state) noexcept
{
  float* __restrict__ hs = h.data();
  if (reset_state)
    std::fill_n(hs, d_inner * d_state, 0.0F);

  for (std::size_t t{0uz}; t < length; ++t) {
    const float* __restrict__ dt = delta.data() + (t * d_inner);
    const float* __restrict__ ut = u.data() + (t * d_inner);
    const float* __restrict__ bt = b.data() + (t * d_state);
    const float* __restrict__ c_t = c_proj.data() + (t * d_state);
    const float* __restrict__ dk = d_skip.data();
    float* __restrict__ y_t = y.data() + (t * d_inner);

    std::size_t c0{0uz};
    for (; c0 + 16uz <= d_inner; c0 += 16uz)
      _mm512_storeu_ps(y_t + c0, _mm512_mul_ps(_mm512_loadu_ps(dk + c0), _mm512_loadu_ps(ut + c0)));
    for (; c0 < d_inner; ++c0)
      y_t[c0] = dk[c0] * ut[c0]; // skip term

    for (std::size_t n{0uz}; n < d_state; ++n) {
      float* __restrict__ hn = hs + (n * d_inner);
      const float* __restrict__ an = a_neg.data() + (n * d_inner);
      const float bn = bt[n];
      const __m512 vbn = _mm512_set1_ps(bn);
      const float cn = c_t[n];
      const __m512 vcn = _mm512_set1_ps(cn);

      std::size_t c{0uz};
      for (; c + 16uz <= d_inner; c += 16uz) {
        const __m512 vdt = _mm512_loadu_ps(dt + c);
        const __m512 vut = _mm512_loadu_ps(ut + c);
        const __m512 van = _mm512_loadu_ps(an + c);
        __m512 vhn = _mm512_loadu_ps(hn + c);

        const __m512 da_nc = exp16(_mm512_mul_ps(vdt, van));
        const __m512 dbu_nc = _mm512_mul_ps(_mm512_mul_ps(vdt, vbn), vut);

        vhn = _mm512_fmadd_ps(da_nc, vhn, dbu_nc);
        _mm512_storeu_ps(hn + c, vhn);
        _mm512_storeu_ps(y_t + c, _mm512_fmadd_ps(vhn, vcn, _mm512_loadu_ps(y_t + c)));
      }
      for (; c < d_inner; ++c) {
        const float da_nc = std::exp(dt[c] * an[c]);
        const float dbu_nc = dt[c] * bn * ut[c];
        hn[c] = (da_nc * hn[c]) + dbu_nc;
        y_t[c] += hn[c] * cn;
      }
    }
  }
}

} // namespace fe
