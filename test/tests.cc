#include "arena/arena.h"
#include "arena/thread_pool.h"
#include "heads/flow/flow.h"
#include "kernels/kernels.h"
#include "loader/safetensors.h"
#include "models/mamba/mamba.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <span>
#include <thread>
#include <vector>

namespace {

std::vector<float> seq(std::size_t n, float a, float b)
{
  std::vector<float> v(n);
  for (std::size_t i{0uz}; i < n; ++i)
    v[i] = std::sin(static_cast<float>(i) * a + b); // deterministic, mixed-sign
  return v;
}

// convert a float vector to BF16
std::vector<uint16_t> to_bf16(const std::vector<float>& v)
{
  std::vector<uint16_t> out(v.size());
  for (std::size_t i{0uz}; i < v.size(); ++i) {
    uint32_t bits;
    std::memcpy(&bits, &v[i], sizeof(bits));
    out[i] = static_cast<uint16_t>(bits >> 16);
  }
  return out;
}

constexpr float kTol = 2e-3F; // ~1 ULP exp8/log8 + fp32 reduction reorder

} // namespace

TEST(Matmul, MatchesNaive)
{
  for (const std::size_t rows : {1uz, 4uz, 7uz}) { // 1 hits the single-vector fast path
    const std::size_t in{37uz}; // exercises SIMD tails without assuming an aligned width
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

TEST(ThreadPool, ParallelForCompletesAllSlices)
{
  std::vector<fe::Task> ring(8uz);
  std::vector<std::size_t> sequence(8uz);
  std::vector<std::jthread> workers(4uz);
  fe::ThreadPool pool{ring, sequence, workers, 4u};
  std::array<std::size_t, 64uz> counts{};

  fe::parallel_for(pool, counts.size(), [&](std::size_t lo, std::size_t hi) noexcept {
    for (std::size_t i{lo}; i < hi; ++i)
      counts[i] = i + 1uz;
  });

  for (std::size_t i{0uz}; i < counts.size(); ++i)
    EXPECT_EQ(counts[i], i + 1uz);
}

TEST(ThreadPool, WakesAfterIdlePark)
{
  using namespace std::chrono_literals;
  std::vector<fe::Task> ring(8uz);
  std::vector<std::size_t> sequence(8uz);
  std::vector<std::jthread> workers(2uz);
  fe::ThreadPool pool{ring, sequence, workers, 2u};
  std::this_thread::sleep_for(10ms);

  std::array<std::size_t, 32uz> values{};
  fe::parallel_for(pool, values.size(), [&](std::size_t lo, std::size_t hi) noexcept {
    for (std::size_t i{lo}; i < hi; ++i)
      values[i] = 3uz * i;
  });
  for (std::size_t i{0uz}; i < values.size(); ++i)
    EXPECT_EQ(values[i], 3uz * i);
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

TEST(DiscretizeAndScan, MatchesNaiveRecurrence)
{
  const std::size_t l{3uz};
  const std::size_t di{5uz};
  const std::size_t ds{2uz};
  const std::vector<float> delta = seq(l * di, 0.1F, 0.4F);
  const std::vector<float> a_log = seq(di * ds, 0.2F, 0.1F);
  const std::vector<float> b = seq(l * ds, 0.3F, 0.7F);
  const std::vector<float> u = seq(l * di, 0.25F, 0.9F);
  const std::vector<float> cp = seq(l * ds, 0.35F, 0.15F);
  const std::vector<float> d = seq(di, 0.15F, 0.2F);
  std::vector<float> a_neg(ds * di);
  for (std::size_t nn{0uz}; nn < ds; ++nn)
    for (std::size_t c{0uz}; c < di; ++c)
      a_neg[(nn * di) + c] = -std::exp(a_log[(c * ds) + nn]);
  std::vector<float> h(ds * di);
  std::vector<float> y(l * di);

  fe::discretize_and_scan(delta, a_neg, b, u, cp, d, h, y, l, di, ds, true);

  std::vector<float> hr(ds * di, 0.0F);
  std::vector<float> yr(l * di);
  for (std::size_t t{0uz}; t < l; ++t) {
    for (std::size_t c{0uz}; c < di; ++c)
      yr[(t * di) + c] = d[c] * u[(t * di) + c];
    for (std::size_t nn{0uz}; nn < ds; ++nn) {
      for (std::size_t c{0uz}; c < di; ++c) {
        const float da = std::exp(delta[(t * di) + c] * -std::exp(a_log[(c * ds) + nn]));
        const float dbu = delta[(t * di) + c] * b[(t * ds) + nn] * u[(t * di) + c];
        hr[(nn * di) + c] = (da * hr[(nn * di) + c]) + dbu;
        yr[(t * di) + c] += hr[(nn * di) + c] * cp[(t * ds) + nn];
      }
    }
  }
  for (std::size_t i{0uz}; i < l * di; ++i)
    EXPECT_NEAR(y[i], yr[i], kTol);
}

TEST(DiscretizeAndScan, PreservesStreamingState)
{
  constexpr std::size_t di{3uz};
  constexpr std::size_t ds{2uz};
  const std::vector<float> delta{0.2F, 0.3F, 0.4F};
  const std::vector<float> a_neg{-1.0F, -2.0F, -3.0F, -0.5F, -1.5F, -2.5F};
  const std::vector<float> b{0.7F, -0.2F};
  const std::vector<float> u{0.4F, -0.6F, 0.8F};
  const std::vector<float> cp{0.9F, 0.25F};
  const std::vector<float> d{0.1F, 0.2F, 0.3F};
  std::vector<float> h(ds * di);
  std::vector<float> first(di), second(di);

  fe::discretize_and_scan(delta, a_neg, b, u, cp, d, h, first, 1uz, di, ds, true);
  const std::vector<float> state_after_first = h;
  fe::discretize_and_scan(delta, a_neg, b, u, cp, d, h, second, 1uz, di, ds, false);

  for (std::size_t i{0uz}; i < h.size(); ++i)
    EXPECT_NE(h[i], state_after_first[i]);
  for (std::size_t c{0uz}; c < di; ++c)
    EXPECT_NE(second[c], first[c]);
}

namespace {

// Build a small in-memory flow head (random weights) for invariant tests
struct FlowFixture
{
  static constexpr std::size_t kA = 8uz, kC = 32uz, kH = 48uz, kT = 16uz, kL = 2uz;
  // weights stored as BF16
  std::vector<uint16_t> in_ = to_bf16(seq(kH * kA, 0.11F, 0.0F));
  std::vector<uint16_t> tp_ = to_bf16(seq(kH * kT, 0.13F, 1.0F));
  std::vector<uint16_t> cp_ = to_bf16(seq(kH * kC, 0.09F, 0.5F));
  std::vector<uint16_t> op_ = to_bf16(seq(kA * kH, 0.07F, 0.2F));
  std::vector<uint16_t> l0_ = to_bf16(seq(kH * kH, 0.05F, 0.1F));
  std::vector<uint16_t> l1_ = to_bf16(seq(kH * kH, 0.06F, 0.3F));
  std::array<std::vector<uint16_t>, kL> layers_{l0_, l1_};
  std::vector<std::byte> slab = std::vector<std::byte>(1uz << 20);
  fe::Arena arena{std::span<std::byte>{slab}};

  static fe::TensorView view(const char* name, uint16_t* data, std::size_t r, std::size_t c)
  {
    fe::TensorView v{};
    v.data = data;
    v.dtype = fe::TensorView::Dtype::BF16;
    v.shape[0] = r;
    v.shape[1] = c;
    v.ndim = 2;
    std::size_t j{0uz};
    for (const char* p = name; (*p != '\0') && (j + 1uz < v.name.size()); ++p)
      v.name[j++] = *p;
    return v;
  }

  static fe::TensorView view_f32(const char* name, float* data, std::size_t r, std::size_t c)
  {
    fe::TensorView v{};
    v.data = data;
    v.dtype = fe::TensorView::Dtype::F32;
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

struct FlowF32Fixture
{
  static constexpr std::size_t kA = FlowFixture::kA;
  static constexpr std::size_t kC = FlowFixture::kC;
  static constexpr std::size_t kH = FlowFixture::kH;
  static constexpr std::size_t kT = FlowFixture::kT;
  std::vector<float> in_ = seq(kH * kA, 0.011F, 0.0F);
  std::vector<float> tp_ = seq(kH * kT, 0.013F, 1.0F);
  std::vector<float> cp_ = seq(kH * kC, 0.009F, 0.5F);
  std::vector<float> op_ = seq(kA * kH, 0.007F, 0.2F);
  std::vector<std::byte> slab = std::vector<std::byte>(1uz << 20);
  fe::Arena arena{std::span<std::byte>{slab}};

  std::vector<fe::TensorView> views()
  {
    std::vector<fe::TensorView> v;
    v.push_back(FlowFixture::view_f32("flow.in_proj.weight", in_.data(), kH, kA));
    v.push_back(FlowFixture::view_f32("flow.time_proj.weight", tp_.data(), kH, kT));
    v.push_back(FlowFixture::view_f32("flow.cond_proj.weight", cp_.data(), kH, kC));
    v.push_back(FlowFixture::view_f32("flow.out_proj.weight", op_.data(), kA, kH));
    return v;
  }
};

struct MambaFixture
{
  static constexpr std::size_t kDm = 4uz, kDi = 4uz, kDs = 2uz, kDc = 3uz, kDr = 1uz, kVocab = 8uz;
  std::vector<float> embeddings = seq(kVocab * kDm, 0.07F, 0.1F);
  std::vector<float> norm_f = std::vector<float>(kDm, 1.0F);
  std::vector<float> norm = std::vector<float>(kDm, 1.0F);
  std::vector<float> in_proj = seq(2uz * kDi * kDm, 0.03F, 0.1F);
  std::vector<float> conv_w = seq(kDi * kDc, 0.04F, 0.2F);
  std::vector<float> conv_b = seq(kDi, 0.02F, 0.0F);
  std::vector<float> x_proj = seq((kDr + (2uz * kDs)) * kDi, 0.02F, 0.3F);
  std::vector<float> dt_w = seq(kDi * kDr, 0.015F, 0.4F);
  std::vector<float> dt_b = std::vector<float>(kDi, 0.1F);
  std::vector<float> a_log = seq(kDi * kDs, 0.025F, -0.2F);
  std::vector<float> d = std::vector<float>(kDi, 0.2F);
  std::vector<float> out_proj = seq(kDm * kDi, 0.02F, 0.5F);
  std::vector<std::byte> slab = std::vector<std::byte>(1uz << 20);
  fe::Arena arena{std::span<std::byte>{slab}};

  static fe::TensorView view(const char* name, float* data,
                             std::initializer_list<std::size_t> shape)
  {
    fe::TensorView v{};
    v.data = data;
    v.dtype = fe::TensorView::Dtype::F32;
    v.ndim = shape.size();
    std::size_t axis{0uz};
    for (const std::size_t extent : shape)
      v.shape[axis++] = extent;
    std::size_t j{0uz};
    for (const char* p = name; (*p != '\0') && (j + 1uz < v.name.size()); ++p)
      v.name[j++] = *p;
    return v;
  }

  std::vector<fe::TensorView> views()
  {
    return {
        view("backbone.embeddings.weight", embeddings.data(), {kVocab, kDm}),
        view("backbone.norm_f.weight", norm_f.data(), {kDm}),
        view("backbone.layers.0.norm.weight", norm.data(), {kDm}),
        view("backbone.layers.0.mixer.in_proj.weight", in_proj.data(), {2uz * kDi, kDm}),
        view("backbone.layers.0.mixer.conv1d.weight", conv_w.data(), {kDi, 1uz, kDc}),
        view("backbone.layers.0.mixer.conv1d.bias", conv_b.data(), {kDi}),
        view("backbone.layers.0.mixer.x_proj.weight", x_proj.data(), {kDr + (2uz * kDs), kDi}),
        view("backbone.layers.0.mixer.dt_proj.weight", dt_w.data(), {kDi, kDr}),
        view("backbone.layers.0.mixer.dt_proj.bias", dt_b.data(), {kDi}),
        view("backbone.layers.0.mixer.A_log", a_log.data(), {kDi, kDs}),
        view("backbone.layers.0.mixer.D", d.data(), {kDi}),
        view("backbone.layers.0.mixer.out_proj.weight", out_proj.data(), {kDm, kDi}),
    };
  }
};

} // namespace

TEST(Mamba, StreamingStateCanBeSnapshottedAndRestored)
{
  MambaFixture fx;
  fe::Mamba model{fx.views(), fx.arena};
  ASSERT_TRUE(model.valid());
  std::vector<float> state(model.state_size(), 0.0F);
  const std::vector<float> first = seq(MambaFixture::kDm, 0.2F, 0.1F);
  const std::vector<float> second = seq(MambaFixture::kDm, 0.3F, -0.2F);
  std::vector<float> ignored(MambaFixture::kDm), branch_a(MambaFixture::kDm),
      branch_b(MambaFixture::kDm), fresh_out(MambaFixture::kDm);

  model.decode(first, state, ignored);
  const std::vector<float> snapshot = state;
  model.decode(second, state, branch_a);
  state = snapshot;
  model.decode(second, state, branch_b);

  for (std::size_t i{0uz}; i < branch_a.size(); ++i)
    EXPECT_EQ(branch_a[i], branch_b[i]);

  std::vector<float> fresh_state(model.state_size(), 0.0F);
  model.decode(second, fresh_state, fresh_out);
  float continuation_delta{0.0F};
  for (std::size_t i{0uz}; i < branch_a.size(); ++i)
    continuation_delta += std::fabs(branch_a[i] - fresh_out[i]);
  EXPECT_GT(continuation_delta, 0.0F);
}

TEST(Mamba, StreamingMatchesBatchForward)
{
  MambaFixture fx;
  fe::Mamba model{fx.views(), fx.arena};
  ASSERT_TRUE(model.valid());
  constexpr std::size_t kLength{4uz};
  const std::vector<float> input = seq(kLength * MambaFixture::kDm, 0.17F, -0.3F);
  std::vector<float> batch(input.size()), streamed(input.size());
  std::vector<float> state(model.state_size(), 0.0F);

  model.forward(input, batch, kLength);
  for (std::size_t t{0uz}; t < kLength; ++t)
    model.decode({input.data() + (t * MambaFixture::kDm), MambaFixture::kDm}, state,
                 {streamed.data() + (t * MambaFixture::kDm), MambaFixture::kDm});

  for (std::size_t i{0uz}; i < batch.size(); ++i)
    EXPECT_NEAR(streamed[i], batch[i], 3.0e-6F);
}

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

TEST(FlowHead, AcceptsF32Weights)
{
  FlowF32Fixture fx;
  fe::FlowHead head{fx.views(), fx.arena};
  ASSERT_TRUE(head.valid());
  const std::vector<float> cond = seq(FlowF32Fixture::kC, 0.1F, 0.0F);
  const std::vector<float> x0 = seq(FlowF32Fixture::kA, 0.3F, 0.2F);
  std::vector<float> out(FlowF32Fixture::kA);
  head.sample(cond, x0, 2uz, fe::FlowHead::kEuler, out);
  for (float v : out)
    EXPECT_TRUE(std::isfinite(v));
}

TEST(FlowHead, ResumableSamplerMatchesMonolithicResult)
{
  FlowFixture fx;
  fe::FlowHead head{fx.views(), fx.arena};
  ASSERT_TRUE(head.valid());
  const std::vector<float> cond = seq(FlowFixture::kC, 0.1F, 0.0F);
  const std::vector<float> x0 = seq(FlowFixture::kA, 0.3F, 0.2F);
  std::vector<float> reference(FlowFixture::kA), resumed(FlowFixture::kA);
  head.sample(cond, x0, 10uz, fe::FlowHead::kHeun, reference);

  std::vector<float> workspace(head.sampler_workspace_size());
  fe::FlowHead::SamplerState state{};
  ASSERT_TRUE(head.sampler_begin(cond, x0, 10uz, fe::FlowHead::kHeun, workspace, state));
  EXPECT_EQ(head.sampler_advance(state, 3uz, resumed), 3uz);
  EXPECT_EQ(state.remaining(), 7uz);
  EXPECT_EQ(head.sampler_advance(state, 4uz, resumed), 4uz);
  EXPECT_EQ(state.remaining(), 3uz);
  EXPECT_EQ(head.sampler_advance(state, 99uz, resumed), 3uz);
  EXPECT_EQ(state.remaining(), 0uz);

  for (std::size_t i{0uz}; i < resumed.size(); ++i)
    EXPECT_EQ(resumed[i], reference[i]);
}

TEST(FlowHead, ResumableSamplerRejectsSmallWorkspace)
{
  FlowFixture fx;
  fe::FlowHead head{fx.views(), fx.arena};
  ASSERT_TRUE(head.valid());
  const std::vector<float> cond(FlowFixture::kC);
  const std::vector<float> x0(FlowFixture::kA);
  std::vector<float> workspace(head.sampler_workspace_size() - 1uz);
  fe::FlowHead::SamplerState state{};
  EXPECT_FALSE(head.sampler_begin(cond, x0, 2uz, fe::FlowHead::kEuler, workspace, state));
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
  // damp: scale BF16 weights by 0.1 to keep activations from blowing up under RK4
  auto damp = [](std::vector<uint16_t>& w) {
    for (uint16_t& e : w) {
      uint32_t bits = static_cast<uint32_t>(e) << 16;
      float f;
      std::memcpy(&f, &bits, sizeof(f));
      f *= 0.1F;
      std::memcpy(&bits, &f, sizeof(f));
      e = static_cast<uint16_t>(bits >> 16);
    }
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
  fx.tp_ = to_bf16(seq(FlowFixture::kH * 15uz, 0.13F, 1.0F)); // odd time_dim
  auto v = fx.views();
  v[1] = FlowFixture::view("flow.time_proj.weight", fx.tp_.data(), FlowFixture::kH, 15uz);
  const fe::FlowHead head{v, fx.arena};
  EXPECT_FALSE(head.valid());
}

TEST(FlowHead, RejectsIncompatibleProjectionShape)
{
  FlowFixture fx;
  auto v = fx.views();
  v[3] = FlowFixture::view("flow.out_proj.weight", fx.op_.data(), FlowFixture::kA - 1uz,
                           FlowFixture::kH);
  const fe::FlowHead head{v, fx.arena};
  EXPECT_FALSE(head.valid());
}
