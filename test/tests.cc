#include "arena/arena.h"
#include "heads/flow/flow.h"
#include "kernels/kernels.h"
#include "loader/safetensors.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <span>
#include <vector>

namespace {

std::vector<float> seq(std::size_t n, float a, float b)
{
  std::vector<float> v(n);
  for (std::size_t i{0uz}; i < n; ++i)
    v[i] = std::sin(static_cast<float>(i) * a + b); // deterministic, mixed-sign
  return v;
}

constexpr float kTol = 2e-3F; // ~1 ULP exp8/log8 + fp32 reduction reorder

} // namespace

TEST(Matmul, MatchesNaive)
{
  for (const std::size_t rows : {1uz, 4uz, 7uz}) { // 1 hits the single-vector fast path
    const std::size_t in{40uz};
    const std::size_t out{17uz};
    const std::vector<float> a = seq(rows * in, 0.1F, 0.0F);
    const std::vector<float> w = seq(out * in, 0.07F, 1.0F);
    std::vector<float> got(rows * out);
    fe::matmul(a, w, got, rows, in, out);
    for (std::size_t r{0uz}; r < rows; ++r)
      for (std::size_t o{0uz}; o < out; ++o) {
        float acc{0.0F};
        for (std::size_t k{0uz}; k < in; ++k)
          acc += a[(r * in) + k] * w[(o * in) + k];
        EXPECT_NEAR(got[(r * out) + o], acc, kTol) << "rows=" << rows;
      }
  }
}

TEST(Silu, MatchesReference)
{
  std::vector<float> x = seq(37uz, 0.3F, -2.0F); // >8 to exercise the SIMD body + tail
  const std::vector<float> in = x;
  fe::silu(x);
  for (std::size_t i{0uz}; i < x.size(); ++i)
    EXPECT_NEAR(x[i], in[i] / (1.0F + std::exp(-in[i])), kTol);
}

TEST(Softplus, MatchesReference)
{
  std::vector<float> x = seq(37uz, 0.25F, 0.5F);
  const std::vector<float> in = x;
  fe::softplus(x);
  for (std::size_t i{0uz}; i < x.size(); ++i)
    EXPECT_NEAR(x[i], std::log1p(std::exp(in[i])), kTol);
}

TEST(RmsNorm, NormalizesRows)
{
  const std::size_t rows{3uz};
  const std::size_t dim{20uz};
  const std::vector<float> in = seq(rows * dim, 0.2F, 0.3F);
  const std::vector<float> w(dim, 1.0F);
  std::vector<float> out(rows * dim);
  fe::rmsnorm(in, w, out, rows, dim);
  for (std::size_t r{0uz}; r < rows; ++r) {
    float ss{0.0F};
    for (std::size_t i{0uz}; i < dim; ++i)
      ss += in[(r * dim) + i] * in[(r * dim) + i];
    const float scale = 1.0F / std::sqrt((ss / static_cast<float>(dim)) + 1e-5F);
    for (std::size_t i{0uz}; i < dim; ++i)
      EXPECT_NEAR(out[(r * dim) + i], in[(r * dim) + i] * scale, kTol);
  }
}

TEST(Conv1dCausal, MatchesNaive)
{
  const std::size_t ch{3uz};
  const std::size_t len{6uz};
  const std::size_t k{4uz};
  const std::vector<float> x = seq(ch * len, 0.4F, 0.0F);
  const std::vector<float> w = seq(ch * k, 0.5F, 1.0F);
  const std::vector<float> b = seq(ch, 0.9F, 0.2F);
  std::vector<float> y(ch * len);
  fe::conv1d_causal(x, w, b, y, ch, len, k);
  for (std::size_t c{0uz}; c < ch; ++c)
    for (std::size_t t{0uz}; t < len; ++t) {
      float acc = b[c];
      for (std::size_t j{0uz}; j < k; ++j) {
        const std::size_t start = (k - 1uz) - j;
        if (t >= start)
          acc += w[(c * k) + j] * x[(c * len) + (t - start)];
      }
      EXPECT_NEAR(y[(c * len) + t], acc, kTol);
    }
}

TEST(SelectiveScan, MatchesNaiveRecurrence)
{
  const std::size_t l{3uz};
  const std::size_t di{5uz};
  const std::size_t ds{2uz};
  const std::vector<float> da = seq(l * ds * di, 0.1F, 0.4F);
  const std::vector<float> dbu = seq(l * ds * di, 0.2F, 0.1F);
  const std::vector<float> cp = seq(l * ds, 0.3F, 0.7F);
  const std::vector<float> d = seq(di, 0.15F, 0.2F);
  const std::vector<float> u = seq(l * di, 0.25F, 0.9F);
  std::vector<float> h(ds * di);
  std::vector<float> y(l * di);
  fe::selective_scan(da, dbu, cp, d, u, h, y, l, di, ds);

  std::vector<float> hr(ds * di, 0.0F);
  std::vector<float> yr(l * di);
  for (std::size_t t{0uz}; t < l; ++t) {
    for (std::size_t c{0uz}; c < di; ++c)
      yr[(t * di) + c] = d[c] * u[(t * di) + c];
    for (std::size_t nn{0uz}; nn < ds; ++nn)
      for (std::size_t c{0uz}; c < di; ++c) {
        const std::size_t idx = (((t * ds) + nn) * di) + c;
        hr[(nn * di) + c] = (da[idx] * hr[(nn * di) + c]) + dbu[idx];
        yr[(t * di) + c] += hr[(nn * di) + c] * cp[(t * ds) + nn];
      }
  }
  for (std::size_t i{0uz}; i < l * di; ++i)
    EXPECT_NEAR(y[i], yr[i], kTol);
}

namespace {

// Build a small in-memory flow head (random weights) for invariant tests
struct FlowFixture
{
  static constexpr std::size_t kA = 8uz, kC = 32uz, kH = 48uz, kT = 16uz, kL = 2uz;
  std::vector<float> in_ = seq(kH * kA, 0.11F, 0.0F);
  std::vector<float> tp_ = seq(kH * kT, 0.13F, 1.0F);
  std::vector<float> cp_ = seq(kH * kC, 0.09F, 0.5F);
  std::vector<float> op_ = seq(kA * kH, 0.07F, 0.2F);
  std::vector<float> l0_ = seq(kH * kH, 0.05F, 0.1F);
  std::vector<float> l1_ = seq(kH * kH, 0.06F, 0.3F);
  std::array<std::vector<float>, kL> layers_{l0_, l1_};
  std::vector<std::byte> slab = std::vector<std::byte>(1uz << 20);
  fe::Arena arena{std::span<std::byte>{slab}};

  static fe::TensorView view(const char* name, float* data, std::size_t r, std::size_t c)
  {
    fe::TensorView v{};
    v.data = data;
    v.shape[0] = r;
    v.shape[1] = c;
    v.ndim = 2;
    std::size_t j{0uz};
    for (const char* p = name; (*p != '\0') && (j + 1uz < v.name.size()); ++p)
      v.name[j++] = *p;
    return v;
  }

  std::vector<fe::TensorView> views()
  {
    std::vector<fe::TensorView> v;
    v.push_back(view("flow.in_proj.weight", in_.data(), kH, kA));
    v.push_back(view("flow.time_proj.weight", tp_.data(), kH, kT));
    v.push_back(view("flow.cond_proj.weight", cp_.data(), kH, kC));
    v.push_back(view("flow.out_proj.weight", op_.data(), kA, kH));
    v.push_back(view("flow.layers.0.weight", layers_[0].data(), kH, kH));
    v.push_back(view("flow.layers.1.weight", layers_[1].data(), kH, kH));
    return v;
  }
};

} // namespace

TEST(FlowHead, DeterministicAndFinite)
{
  FlowFixture fx;
  fe::FlowHead head{fx.views(), fx.arena};
  ASSERT_TRUE(head.valid());
  const std::vector<float> cond = seq(FlowFixture::kC, 0.1F, 0.0F);
  const std::vector<float> x0 = seq(FlowFixture::kA, 0.3F, 0.2F);
  std::vector<float> a(FlowFixture::kA), b(FlowFixture::kA);
  head.sample(cond, x0, 10uz, fe::FlowHead::kEuler, a);
  head.sample(cond, x0, 10uz, fe::FlowHead::kEuler, b);
  for (std::size_t i{0uz}; i < a.size(); ++i) {
    EXPECT_EQ(a[i], b[i]) << "not deterministic"; // same inputs -> identical bits
    EXPECT_TRUE(std::isfinite(a[i]));
  }
}

TEST(FlowHead, EulerDiffersFromHeun)
{
  FlowFixture fx;
  fe::FlowHead head{fx.views(), fx.arena};
  ASSERT_TRUE(head.valid());
  const std::vector<float> cond = seq(FlowFixture::kC, 0.1F, 0.0F);
  const std::vector<float> x0 = seq(FlowFixture::kA, 0.3F, 0.2F);
  std::vector<float> e(FlowFixture::kA), h(FlowFixture::kA);
  head.sample(cond, x0, 10uz, fe::FlowHead::kEuler, e);
  head.sample(cond, x0, 10uz, fe::FlowHead::kHeun, h);
  float diff{0.0F};
  for (std::size_t i{0uz}; i < e.size(); ++i)
    diff += std::fabs(e[i] - h[i]);
  EXPECT_GT(diff, 0.0F); // 2nd-order integrator takes a different path
}

TEST(FlowHead, RK4DiffersFromEulerAndFinite)
{
  FlowFixture fx;
  auto damp = [](std::vector<float>& w) {
    for (float& e : w)
      e *= 0.1F;
  };
  damp(fx.in_);
  damp(fx.tp_);
  damp(fx.cp_);
  damp(fx.op_);
  damp(fx.layers_[0]);
  damp(fx.layers_[1]);
  fe::FlowHead head{fx.views(), fx.arena};
  ASSERT_TRUE(head.valid());
  const std::vector<float> cond = seq(FlowFixture::kC, 0.1F, 0.0F);
  const std::vector<float> x0 = seq(FlowFixture::kA, 0.3F, 0.2F);
  std::vector<float> e(FlowFixture::kA), r(FlowFixture::kA);
  head.sample(cond, x0, 10uz, fe::FlowHead::kEuler, e);
  head.sample(cond, x0, 10uz, fe::FlowHead::kRK4, r);
  float diff{0.0F};
  for (std::size_t i{0uz}; i < r.size(); ++i) {
    EXPECT_TRUE(std::isfinite(r[i]));
    diff += std::fabs(e[i] - r[i]);
  }
  EXPECT_GT(diff, 0.0F); // 4th-order path differs from 1st-order Euler
}

TEST(FlowHead, RejectsOddTimeDim)
{
  FlowFixture fx;
  fx.tp_ = seq(FlowFixture::kH * 15uz, 0.13F, 1.0F); // odd time_dim: no sin/cos pairing
  auto v = fx.views();
  v[1] = FlowFixture::view("flow.time_proj.weight", fx.tp_.data(), FlowFixture::kH, 15uz);
  const fe::FlowHead head{v, fx.arena};
  EXPECT_FALSE(head.valid());
}
