#include "kernels/cpu/cephes.h"
#include "kernels/kernels.h"

#include <arm_neon.h>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace fe {
namespace {

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

// 1 recurrence step, state-major [n][c]: advance h in place, emit y (skip term + Σ_n h·c)
inline void scan_advance(const float* __restrict__ da_t, const float* __restrict__ dbu_t,
                         const float* __restrict__ c_t, const float* __restrict__ u_t,
                         const float* __restrict__ d, float* __restrict__ hs,
                         float* __restrict__ y_t, std::size_t d_inner, std::size_t d_state) noexcept
{
  std::size_t c0{0uz};
  for (; c0 + 4uz <= d_inner; c0 += 4uz)
    vst1q_f32(y_t + c0, vmulq_f32(vld1q_f32(d + c0), vld1q_f32(u_t + c0)));
  for (; c0 < d_inner; ++c0)
    y_t[c0] = d[c0] * u_t[c0]; // skip term, scan accumulates onto it

  for (std::size_t n{0uz}; n < d_state; ++n) {
    float* __restrict__ hn = hs + (n * d_inner);
    const float* __restrict__ da_n = da_t + (n * d_inner);
    const float* __restrict__ dbu_n = dbu_t + (n * d_inner);
    const float cn = c_t[n];
    const float32x4_t vcn = vdupq_n_f32(cn);
    std::size_t c{0uz};
    for (; c + 4uz <= d_inner; c += 4uz) {
      float32x4_t vh = vld1q_f32(hn + c);
      vh = vfmaq_f32(vld1q_f32(dbu_n + c), vld1q_f32(da_n + c), vh);
      vst1q_f32(hn + c, vh);
      vst1q_f32(y_t + c, vfmaq_f32(vld1q_f32(y_t + c), vh, vcn));
    }
    for (; c < d_inner; ++c) {
      hn[c] = (da_n[c] * hn[c]) + dbu_n[c];
      y_t[c] += hn[c] * cn;
    }
  }
}

// widen 4 BF16 → 4 F32
inline float32x4_t load_bf16_4(const uint16_t* p) noexcept
{
  return vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vld1_u16(p)), 16));
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
  const float32x4_t one = vdupq_n_f32(1.0F);
  for (; i + 4uz <= out.size(); i += 4uz) {
    const float32x4_t vg = vld1q_f32(g.data() + i);
    const float32x4_t sig = vdivq_f32(vg, vaddq_f32(one, exp4(vnegq_f32(vg))));
    vst1q_f32(out.data() + i, vmulq_f32(vld1q_f32(a.data() + i), sig));
  }
  for (; i < out.size(); ++i) {
    const float v = g[i];
    out[i] = a[i] * (v / (1.0F + std::exp(-v)));
  }
}

void silu(std::span<float> x) noexcept
{
  std::size_t i{0uz};
  const float32x4_t one = vdupq_n_f32(1.0F);
  for (; i + 4uz <= x.size(); i += 4uz) {
    const float32x4_t v = vld1q_f32(x.data() + i);
    vst1q_f32(x.data() + i, vdivq_f32(v, vaddq_f32(one, exp4(vnegq_f32(v)))));
  }
  for (; i < x.size(); ++i)
    x[i] = x[i] / (1.0F + std::exp(-x[i]));
}

void softplus(std::span<float> x) noexcept
{
  // max(v,0) + log(1 + e^-|v|)
  // e^-|v| in (0,1]
  std::size_t i{0uz};
  const float32x4_t one = vdupq_n_f32(1.0F);
  const float32x4_t zero = vdupq_n_f32(0.0F);
  for (; i + 4uz <= x.size(); i += 4uz) {
    const float32x4_t v = vld1q_f32(x.data() + i);
    const float32x4_t nabs = vnegq_f32(vabsq_f32(v));
    vst1q_f32(x.data() + i, vaddq_f32(vmaxq_f32(v, zero), log4(vaddq_f32(one, exp4(nabs)))));
  }
  for (; i < x.size(); ++i)
    x[i] = std::log1p(std::exp(x[i]));
}

void matmul(std::span<const float> in, std::span<const float> w, std::span<float> out,
            std::size_t rows, std::size_t in_dim, std::size_t out_dim, ThreadPool*) noexcept
{
  if (rows == 1uz) { // single vector
    const float* __restrict__ ir = in.data();
    for (std::size_t o{0uz}; o < out_dim; ++o) {
      const float* __restrict__ wr = w.data() + (o * in_dim);
      float acc{0.0F};
      std::size_t i{0uz};
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

void matmul(std::span<const float> in, std::span<const uint16_t> w, std::span<float> out,
            std::size_t rows, std::size_t in_dim, std::size_t out_dim, ThreadPool*) noexcept
{
  if (rows == 1uz) { // single vector
    const float* __restrict__ ir = in.data();
    for (std::size_t o{0uz}; o < out_dim; ++o) {
      const uint16_t* __restrict__ wr = w.data() + (o * in_dim);
      float acc{0.0F};
      std::size_t i{0uz};
      float32x4_t a0 = vdupq_n_f32(0.0F);
      float32x4_t a1 = a0, a2 = a0, a3 = a0;
      for (; i + 16uz <= in_dim; i += 16uz) {
        a0 = vfmaq_f32(a0, vld1q_f32(ir + i), load_bf16_4(wr + i));
        a1 = vfmaq_f32(a1, vld1q_f32(ir + i + 4uz), load_bf16_4(wr + i + 4uz));
        a2 = vfmaq_f32(a2, vld1q_f32(ir + i + 8uz), load_bf16_4(wr + i + 8uz));
        a3 = vfmaq_f32(a3, vld1q_f32(ir + i + 12uz), load_bf16_4(wr + i + 12uz));
      }
      float32x4_t av = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
      for (; i + 4uz <= in_dim; i += 4uz)
        av = vfmaq_f32(av, vld1q_f32(ir + i), load_bf16_4(wr + i));
      acc = vaddvq_f32(av);
      for (; i < in_dim; ++i)
        acc += ir[i] * bf16_to_f32(wr[i]);
      out.data()[o] = acc;
    }
    return;
  }

  // multi-row
  for (std::size_t o{0uz}; o < out_dim; ++o) {
    const uint16_t* __restrict__ wr = w.data() + (o * in_dim);
    for (std::size_t r0{0uz}; r0 < rows; r0 += 4uz) {
      const std::size_t nr = (rows - r0 < 4uz) ? (rows - r0) : 4uz;
      const float* __restrict__ i0 = in.data() + ((r0 + 0uz) * in_dim);
      const float* __restrict__ i1 = in.data() + ((r0 + (nr > 1uz ? 1uz : 0uz)) * in_dim);
      const float* __restrict__ i2 = in.data() + ((r0 + (nr > 2uz ? 2uz : 0uz)) * in_dim);
      const float* __restrict__ i3 = in.data() + ((r0 + (nr > 3uz ? 3uz : 0uz)) * in_dim);
      float acc0{0.0F}, acc1{0.0F}, acc2{0.0F}, acc3{0.0F};
      std::size_t i{0uz};
      float32x4_t v0 = vdupq_n_f32(0.0F);
      float32x4_t v1 = v0, v2 = v0, v3 = v0;
      for (; i + 4uz <= in_dim; i += 4uz) {
        const float32x4_t wv = load_bf16_4(wr + i); // widen once
        v0 = vfmaq_f32(v0, vld1q_f32(i0 + i), wv);
        v1 = vfmaq_f32(v1, vld1q_f32(i1 + i), wv);
        v2 = vfmaq_f32(v2, vld1q_f32(i2 + i), wv);
        v3 = vfmaq_f32(v3, vld1q_f32(i3 + i), wv);
      }
      acc0 = vaddvq_f32(v0);
      acc1 = vaddvq_f32(v1);
      acc2 = vaddvq_f32(v2);
      acc3 = vaddvq_f32(v3);
      for (; i < in_dim; ++i) {
        const float wv = bf16_to_f32(wr[i]);
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
void conv1d_causal(std::span<const float> x, std::span<const float> weight,
                   std::span<const float> bias, std::span<float> y, std::size_t channels,
                   std::size_t length, std::size_t kernel) noexcept
{
  for (std::size_t c{0uz}; c < channels; ++c) {
    const float* __restrict__ xc = x.data() + (c * length);
    const float* __restrict__ wc = weight.data() + (c * kernel);
    float* __restrict__ yc = y.data() + (c * length);
    const float32x4_t vbc = vdupq_n_f32(bias[c]);
    std::size_t t{0uz};
    for (; t + 4uz <= length; t += 4uz)
      vst1q_f32(yc + t, vbc);
    for (; t < length; ++t)
      yc[t] = bias[c];

    for (std::size_t k{0uz}; k < kernel; ++k) {
      const float32x4_t vwk = vdupq_n_f32(wc[k]);
      const std::size_t start = (kernel - 1uz) - k;
      std::size_t ts = start;
      for (; ts + 4uz <= length; ts += 4uz) {
        vst1q_f32(yc + ts, vmlaq_f32(vld1q_f32(yc + ts), vwk, vld1q_f32(xc + ts - start)));
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
    float32x4_t vss = vdupq_n_f32(0.0F);
    std::size_t i{0uz};
    for (; i + 4uz <= dim; i += 4uz) {
      const float32x4_t v = vld1q_f32(ir + i);
      vss = vmlaq_f32(vss, v, v);
    }
    float ss = vaddvq_f32(vss);
    for (; i < dim; ++i)
      ss += ir[i] * ir[i];
    const float scale = 1.0F / std::sqrt((ss / static_cast<float>(dim)) + eps);
    const float32x4_t vscale = vdupq_n_f32(scale);
    i = 0uz;
    for (; i + 4uz <= dim; i += 4uz)
      vst1q_f32(orow + i,
                vmulq_f32(vscale, vmulq_f32(vld1q_f32(ir + i), vld1q_f32(weight.data() + i))));
    for (; i < dim; ++i)
      orow[i] = ir[i] * scale * weight[i];
  }
}

void discretize_and_scan(std::span<const float> delta, std::span<const float> a_log,
                         std::span<const float> b, std::span<const float> u,
                         std::span<const float> c_proj, std::span<const float> d_skip,
                         std::span<float> h, std::span<float> y, std::span<float> a_work,
                         std::size_t length, std::size_t d_inner, std::size_t d_state) noexcept
{
  float* __restrict__ hs = h.data();
  for (std::size_t i{0uz}; i < d_inner * d_state; ++i)
    hs[i] = 0.0F; // h_0 = 0

  // A = -exp(a_log): transpose into a_work then compute exp4 in-place
  float* __restrict__ a_sm = a_work.data();
  for (std::size_t c{0uz}; c < d_inner; ++c)
    for (std::size_t n{0uz}; n < d_state; ++n)
      a_sm[(n * d_inner) + c] = a_log[(c * d_state) + n];

  std::size_t i{0uz};
  const std::size_t plane = d_inner * d_state;
  for (; i + 4uz <= plane; i += 4uz) {
    vst1q_f32(a_sm + i, vnegq_f32(exp4(vld1q_f32(a_sm + i))));
  }
  for (; i < plane; ++i)
    a_sm[i] = -std::exp(a_sm[i]);

  for (std::size_t t{0uz}; t < length; ++t) {
    const float* __restrict__ dt = delta.data() + (t * d_inner);
    const float* __restrict__ ut = u.data() + (t * d_inner);
    const float* __restrict__ bt = b.data() + (t * d_state);
    const float* __restrict__ c_t = c_proj.data() + (t * d_state);
    const float* __restrict__ dk = d_skip.data();
    float* __restrict__ y_t = y.data() + (t * d_inner);

    std::size_t c0{0uz};
    for (; c0 + 4uz <= d_inner; c0 += 4uz)
      vst1q_f32(y_t + c0, vmulq_f32(vld1q_f32(dk + c0), vld1q_f32(ut + c0)));
    for (; c0 < d_inner; ++c0)
      y_t[c0] = dk[c0] * ut[c0]; // skip term

    for (std::size_t n{0uz}; n < d_state; ++n) {
      float* __restrict__ hn = hs + (n * d_inner);
      const float* __restrict__ an = a_sm + (n * d_inner);
      const float bn = bt[n];
      const float32x4_t vbn = vdupq_n_f32(bn);
      const float cn = c_t[n];
      const float32x4_t vcn = vdupq_n_f32(cn);

      std::size_t c{0uz};
      for (; c + 4uz <= d_inner; c += 4uz) {
        const float32x4_t vdt = vld1q_f32(dt + c);
        const float32x4_t vut = vld1q_f32(ut + c);
        const float32x4_t van = vld1q_f32(an + c);
        float32x4_t vhn = vld1q_f32(hn + c);

        const float32x4_t da_nc = exp4(vmulq_f32(vdt, van));
        const float32x4_t dbu_nc = vmulq_f32(vmulq_f32(vdt, vbn), vut);

        vhn = vfmaq_f32(dbu_nc, da_nc, vhn);
        vst1q_f32(hn + c, vhn);
        vst1q_f32(y_t + c, vfmaq_f32(vld1q_f32(y_t + c), vhn, vcn));
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
