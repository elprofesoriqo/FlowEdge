#include "arena/arena.h"
#include "arena/thread_pool.h"
#include "heads/diffusion/diffusion.h"
#include "heads/flow/flow.h"
#include "kernels/kernels.h"
#include "loader/safetensors.h"
#include "models/mamba/mamba.h"
#include "protocol/model_identity.h"
#include "protocol/snapshot.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <span>
#include <string>
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

TEST(ThreadPool, ParallelForHonorsTaskBudget)
{
  std::vector<fe::Task> ring(8uz);
  std::vector<std::size_t> sequence(8uz);
  std::vector<std::jthread> workers(8uz);
  fe::ThreadPool pool{ring, sequence, workers, 8u};
  std::atomic<unsigned> calls{0u};
  fe::parallel_for(pool, 1024uz, 4u, [&](std::size_t, std::size_t) noexcept {
    calls.fetch_add(1u, std::memory_order_relaxed);
  });
  EXPECT_EQ(calls.load(std::memory_order_relaxed), 4u);
}

TEST(Matmul, AdaptiveTaskSelectionUsesUsefulPowerOfTwoTiers)
{
  using enum fe::MatmulWeightType;
  EXPECT_EQ(fe::matmul_task_count(8u, 4uz, 48uz, 48uz, kF32), 1u);
  EXPECT_EQ(fe::matmul_task_count(8u, 1uz, 768uz, 3072uz, kF32), 2u);
  EXPECT_EQ(fe::matmul_task_count(8u, 4uz, 1536uz, 768uz, kF32), 4u);
  EXPECT_EQ(fe::matmul_task_count(8u, 4uz, 768uz, 3072uz, kF32), 8u);
  EXPECT_EQ(fe::matmul_task_count(8u, 4uz, 1536uz, 768uz, kBF16), 8u);
  EXPECT_EQ(fe::matmul_task_count(2u, 4uz, 768uz, 3072uz, kF32), 2u);
  EXPECT_EQ(fe::matmul_task_count(4u, 4uz, 768uz, 3072uz, kF32), 4u);
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

TEST(Mish, MatchesReference)
{
  std::vector<float> values{-30.0F, -3.0F, -0.25F, 0.0F, 0.5F, 4.0F, 30.0F};
  const std::vector<float> input = values;
  fe::mish(values);
  for (std::size_t i{0uz}; i < values.size(); ++i) {
    const float softplus = input[i] > 20.0F ? input[i] : std::log1p(std::exp(input[i]));
    EXPECT_NEAR(values[i], input[i] * std::tanh(softplus), 2e-6F);
  }
}

TEST(DenseConv1d, SameAndStridedMatchNaive)
{
  constexpr std::size_t in_channels{2uz}, out_channels{3uz}, length{5uz}, kernel{3uz};
  const std::vector<float> input = seq(in_channels * length, 0.31F, -0.2F);
  const std::vector<float> weight = seq(out_channels * in_channels * kernel, 0.17F, 0.4F);
  const std::vector<float> bias = seq(out_channels, 0.23F, -0.1F);
  for (const std::size_t stride : {1uz, 2uz}) {
    const std::size_t output_length = stride == 1uz ? length : 3uz;
    std::vector<float> output(out_channels * output_length);
    fe::conv1d(input, weight, bias, output, in_channels, out_channels, length, output_length,
               kernel, stride, 1uz);
    for (std::size_t oc{0uz}; oc < out_channels; ++oc) {
      for (std::size_t ot{0uz}; ot < output_length; ++ot) {
        float expected = bias[oc];
        for (std::size_t ic{0uz}; ic < in_channels; ++ic)
          for (std::size_t k{0uz}; k < kernel; ++k) {
            const std::ptrdiff_t index = static_cast<std::ptrdiff_t>(ot * stride + k) - 1;
            if (index >= 0 && index < static_cast<std::ptrdiff_t>(length))
              expected += input[(ic * length) + static_cast<std::size_t>(index)] *
                          weight[((oc * in_channels + ic) * kernel) + k];
          }
        EXPECT_NEAR(output[(oc * output_length) + ot], expected, 2e-6F);
      }
    }
  }
}

TEST(ConvTranspose1d, MatchesNaive)
{
  constexpr std::size_t in_channels{2uz}, out_channels{3uz}, input_length{3uz};
  constexpr std::size_t output_length{6uz}, kernel{4uz};
  const std::vector<float> input = seq(in_channels * input_length, 0.37F, 0.2F);
  const std::vector<float> weight = seq(in_channels * out_channels * kernel, 0.19F, -0.3F);
  const std::vector<float> bias = seq(out_channels, 0.13F, 0.1F);
  std::vector<float> output(out_channels * output_length);
  fe::conv_transpose1d(input, weight, bias, output, in_channels, out_channels, input_length,
                       output_length, kernel, 2uz, 1uz);
  std::vector<float> expected(out_channels * output_length);
  for (std::size_t oc{0uz}; oc < out_channels; ++oc)
    std::fill_n(expected.data() + oc * output_length, output_length, bias[oc]);
  for (std::size_t ic{0uz}; ic < in_channels; ++ic)
    for (std::size_t it{0uz}; it < input_length; ++it)
      for (std::size_t oc{0uz}; oc < out_channels; ++oc)
        for (std::size_t k{0uz}; k < kernel; ++k) {
          const std::ptrdiff_t ot = static_cast<std::ptrdiff_t>(it * 2uz + k) - 1;
          if (ot >= 0 && ot < static_cast<std::ptrdiff_t>(output_length))
            expected[(oc * output_length) + static_cast<std::size_t>(ot)] +=
                input[(ic * input_length) + it] * weight[((ic * out_channels + oc) * kernel) + k];
        }
  for (std::size_t i{0uz}; i < output.size(); ++i)
    EXPECT_NEAR(output[i], expected[i], 2e-6F);
}

TEST(DiffusionConvolution, ThreadedSpecializationsMatchCallerThread)
{
  constexpr std::size_t channels{256uz};
  constexpr std::size_t input_length{16uz};
  const std::vector<float> input = seq(channels * input_length, 0.013F, -0.2F);
  const std::vector<float> conv_weight = seq(channels * channels * 5uz, 0.007F, 0.1F);
  const std::vector<float> transpose_weight = seq(channels * channels * 4uz, 0.009F, -0.1F);
  const std::vector<float> bias = seq(channels, 0.03F, 0.0F);
  std::vector<float> expected(channels * input_length), threaded(expected.size());
  std::vector<fe::Task> ring(8uz);
  std::vector<std::size_t> sequence(8uz);
  std::vector<std::jthread> workers(4uz);
  fe::ThreadPool pool{ring, sequence, workers, 4u};

  fe::conv1d(input, conv_weight, bias, expected, channels, channels, input_length, input_length,
             5uz, 1uz, 2uz);
  fe::conv1d(input, conv_weight, bias, threaded, channels, channels, input_length, input_length,
             5uz, 1uz, 2uz, &pool);
  EXPECT_EQ(threaded, expected);

  constexpr std::size_t upsampled_length{32uz};
  expected.resize(channels * upsampled_length);
  threaded.resize(expected.size());
  fe::conv_transpose1d(input, transpose_weight, bias, expected, channels, channels, input_length,
                       upsampled_length, 4uz, 2uz, 1uz);
  fe::conv_transpose1d(input, transpose_weight, bias, threaded, channels, channels, input_length,
                       upsampled_length, 4uz, 2uz, 1uz, &pool);
  EXPECT_EQ(threaded, expected);
}

TEST(GroupNorm, MatchesReference)
{
  constexpr std::size_t channels{4uz}, length{3uz}, groups{2uz};
  std::vector<float> values = seq(channels * length, 0.29F, -0.4F);
  const std::vector<float> input = values;
  const std::vector<float> weight{0.5F, 1.25F, -0.75F, 0.8F};
  const std::vector<float> bias{-0.2F, 0.1F, 0.3F, -0.4F};
  fe::group_norm(values, weight, bias, channels, length, groups);
  for (std::size_t group{0uz}; group < groups; ++group) {
    double mean{0.0};
    double square_sum{0.0};
    for (std::size_t channel{group * 2uz}; channel < (group + 1uz) * 2uz; ++channel)
      for (std::size_t t{0uz}; t < length; ++t) {
        const double value = input[(channel * length) + t];
        mean += value;
        square_sum += value * value;
      }
    mean /= 6.0;
    const double variance = (square_sum / 6.0) - mean * mean;
    for (std::size_t channel{group * 2uz}; channel < (group + 1uz) * 2uz; ++channel)
      for (std::size_t t{0uz}; t < length; ++t) {
        const float expected = static_cast<float>((input[(channel * length) + t] - mean) /
                                                  std::sqrt(variance + 1e-5)) *
                                   weight[channel] +
                               bias[channel];
        EXPECT_NEAR(values[(channel * length) + t], expected, 2e-6F);
      }
  }
}

TEST(Film, AppliesPerChannelScaleAndBias)
{
  std::vector<float> values{1.0F, 2.0F, 3.0F, -1.0F, -2.0F, -3.0F};
  fe::film(values, std::array{2.0F, -0.5F}, std::array{0.25F, 1.0F}, 2uz, 3uz);
  EXPECT_EQ(values, (std::vector<float>{2.25F, 4.25F, 6.25F, 1.5F, 2.0F, 2.5F}));
}

TEST(DiffusionTimestepEmbedding, MatchesLeRobotDefinition)
{
  std::array<float, 8> output{};
  fe::diffusion_timestep_embedding(7.0F, output);
  const float factor = std::log(10000.0F) / 3.0F;
  for (std::size_t i{0uz}; i < 4uz; ++i) {
    const float phase = 7.0F * std::exp(-factor * static_cast<float>(i));
    EXPECT_NEAR(output[i], std::sin(phase), 1e-6F);
    EXPECT_NEAR(output[4uz + i], std::cos(phase), 1e-6F);
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
    v.bytes = r * c * sizeof(std::uint16_t);
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
    v.bytes = r * c * sizeof(float);
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
    v.bytes = sizeof(float);
    for (const std::size_t extent : shape)
      v.bytes *= extent;
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

struct DiffusionFixture
{
  static constexpr std::size_t kActionDim{2uz};
  static constexpr std::size_t kHorizon{4uz};
  static constexpr std::size_t kConditionDim{3uz};
  static constexpr std::size_t kTimeDim{4uz};
  static constexpr std::size_t kKernel{3uz};
  static constexpr std::array<std::size_t, 2> kDims{8uz, 16uz};

  std::vector<std::vector<float>> storage{};
  std::vector<fe::TensorView> tensors{};
  std::vector<std::byte> slab = std::vector<std::byte>(4096uz);
  fe::Arena arena{std::span<std::byte>{slab}};

  DiffusionFixture()
  {
    storage.reserve(128uz);
    tensors.reserve(128uz);
    add("dp.meta", {16uz},
        {1.0F, 2.0F, 4.0F, 2.0F, 2.0F, 3.0F, 2.0F, 3.0F, 2.0F, 4.0F, 10.0F, 0.0F, 0.0F, 1.0F, 1.0F,
         0.0F});
    add("dp.dims", {2uz}, {8.0F, 16.0F});
    add("dp.action_min", {2uz}, {-2.0F, 0.0F});
    add("dp.action_max", {2uz}, {2.0F, 10.0F});
    linear("dp.te1", 16uz, kTimeDim, 0.01F);
    linear("dp.te2", kTimeDim, 16uz, 0.02F);

    residual("dp.d0.r0", kActionDim, kDims[0], 0.03F);
    residual("dp.d0.r1", kDims[0], kDims[0], 0.04F);
    conv("dp.d0.ds", kDims[0], kDims[0], 3uz, 0.05F);
    residual("dp.d1.r0", kDims[0], kDims[1], 0.06F);
    residual("dp.d1.r1", kDims[1], kDims[1], 0.07F);
    residual("dp.m0", kDims[1], kDims[1], 0.08F);
    residual("dp.m1", kDims[1], kDims[1], 0.09F);
    residual("dp.u0.r0", 2uz * kDims[1], kDims[0], 0.10F);
    residual("dp.u0.r1", kDims[0], kDims[0], 0.11F);
    conv("dp.u0.us", kDims[0], kDims[0], 4uz, 0.12F);
    conv("dp.f.c", kDims[0], kDims[0], kKernel, 0.13F);
    norm("dp.f.n", kDims[0]);
    conv("dp.f.o", kActionDim, kDims[0], 1uz, 0.14F);
  }

private:
  std::vector<float> weights(std::size_t count, float phase) const
  {
    std::vector<float> result = seq(count, 0.173F, phase);
    for (float& value : result)
      value *= 0.075F;
    return result;
  }

  void add(const std::string& name, std::initializer_list<std::size_t> shape,
           std::vector<float> values)
  {
    storage.push_back(std::move(values));
    fe::TensorView view{};
    view.data = storage.back().data();
    view.dtype = fe::TensorView::Dtype::F32;
    view.ndim = static_cast<std::uint8_t>(shape.size());
    view.bytes = storage.back().size() * sizeof(float);
    std::size_t axis{0uz};
    for (const std::size_t extent : shape)
      view.shape[axis++] = extent;
    ASSERT_LT(name.size(), view.name.size());
    std::copy(name.begin(), name.end(), view.name.begin());
    tensors.push_back(view);
  }

  void conv(const std::string& prefix, std::size_t out_channels, std::size_t in_channels,
            std::size_t kernel, float phase)
  {
    add(prefix + ".w", {out_channels, in_channels, kernel},
        weights(out_channels * in_channels * kernel, phase));
    add(prefix + ".b", {out_channels}, std::vector<float>(out_channels, phase * 0.01F));
  }

  void linear(const std::string& prefix, std::size_t out_features, std::size_t in_features,
              float phase)
  {
    add(prefix + ".w", {out_features, in_features}, weights(out_features * in_features, phase));
    add(prefix + ".b", {out_features}, std::vector<float>(out_features, phase * 0.01F));
  }

  void norm(const std::string& prefix, std::size_t channels)
  {
    add(prefix + ".w", {channels}, std::vector<float>(channels, 1.0F));
    add(prefix + ".b", {channels}, std::vector<float>(channels, 0.0F));
  }

  void residual(const std::string& prefix, std::size_t in_channels, std::size_t out_channels,
                float phase)
  {
    conv(prefix + ".c1", out_channels, in_channels, kKernel, phase);
    norm(prefix + ".n1", out_channels);
    linear(prefix + ".film", 2uz * out_channels, kTimeDim + kConditionDim, phase + 0.01F);
    conv(prefix + ".c2", out_channels, out_channels, kKernel, phase + 0.02F);
    norm(prefix + ".n2", out_channels);
    if (in_channels != out_channels)
      conv(prefix + ".res", out_channels, in_channels, 1uz, phase + 0.03F);
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

TEST(SnapshotProtocol, RoundTripsAndRejectsMismatchTruncationAndCorruption)
{
  fe::ModelIdentity identity{};
  for (std::size_t i{0uz}; i < identity.digest.size(); ++i)
    identity.digest[i] = static_cast<std::uint8_t>(3uz * i + 1uz);
  identity.architecture = FE_ARCH_MAMBA;
  identity.precision = FE_PRECISION_MIXED;
  identity.d_model = 64u;
  identity.n_layers = 4u;
  identity.d_inner = 128u;
  identity.d_state = 16u;
  identity.d_conv = 4u;

  const std::array<float, 7> state{0.25F, -1.0F, 4.0F, 2.5F, 0.0F, 8.0F, -3.0F};
  std::vector<std::byte> encoded(fe::decode_snapshot_bytes(sizeof(state)));
  ASSERT_TRUE(fe::export_decode_snapshot(identity, std::as_bytes(std::span{state}), encoded));

  std::array<float, state.size()> restored{};
  ASSERT_TRUE(
      fe::import_decode_snapshot(identity, encoded, std::as_writable_bytes(std::span{restored})));
  EXPECT_EQ(restored, state);

  fe::ModelIdentity wrong_model = identity;
  wrong_model.digest[0] ^= 0x80u;
  auto mismatch =
      fe::import_decode_snapshot(wrong_model, encoded, std::as_writable_bytes(std::span{restored}));
  ASSERT_FALSE(mismatch);
  EXPECT_EQ(mismatch.error().code, 9);

  auto truncated =
      fe::import_decode_snapshot(identity, std::span{encoded}.first(encoded.size() - 1uz),
                                 std::as_writable_bytes(std::span{restored}));
  ASSERT_FALSE(truncated);
  EXPECT_EQ(truncated.error().code, 1);

  encoded.back() ^= std::byte{0x01};
  auto corrupted =
      fe::import_decode_snapshot(identity, encoded, std::as_writable_bytes(std::span{restored}));
  ASSERT_FALSE(corrupted);
  EXPECT_EQ(corrupted.error().code, 10);
}

TEST(ModelIdentity, FingerprintIsStableAndContentSensitive)
{
  const std::array<std::byte, 9> first{std::byte{0}, std::byte{1}, std::byte{2},
                                       std::byte{3}, std::byte{4}, std::byte{5},
                                       std::byte{6}, std::byte{7}, std::byte{8}};
  auto changed = first;
  changed[4] ^= std::byte{0x80};
  EXPECT_EQ(fe::fingerprint_bytes(first), fe::fingerprint_bytes(first));
  EXPECT_NE(fe::fingerprint_bytes(first), fe::fingerprint_bytes(changed));
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

TEST(DiffusionHead, DenoiserIsDeterministicFiniteAndConditioned)
{
  DiffusionFixture fixture;
  fe::DiffusionHead head{fixture.tensors, fixture.arena};
  ASSERT_TRUE(head.valid());
  EXPECT_EQ(head.config().action_dim, DiffusionFixture::kActionDim);
  EXPECT_EQ(head.config().horizon, DiffusionFixture::kHorizon);
  EXPECT_EQ(head.config().condition_dim, DiffusionFixture::kConditionDim);
  const std::vector<float> sample = seq(8uz, 0.23F, -0.1F);
  const std::vector<float> condition = seq(3uz, 0.41F, 0.2F);
  std::vector<float> workspace(head.sampler_workspace_size());
  std::vector<float> first(8uz), second(8uz), changed(8uz);
  ASSERT_TRUE(head.denoise(condition, sample, 7.0F, workspace, first));
  ASSERT_TRUE(head.denoise(condition, sample, 7.0F, workspace, second));
  std::vector<float> other_condition = condition;
  other_condition[0] += 0.75F;
  ASSERT_TRUE(head.denoise(other_condition, sample, 7.0F, workspace, changed));
  float condition_delta{0.0F};
  for (std::size_t i{0uz}; i < first.size(); ++i) {
    EXPECT_FLOAT_EQ(first[i], second[i]);
    EXPECT_TRUE(std::isfinite(first[i]));
    condition_delta += std::fabs(first[i] - changed[i]);
  }
  EXPECT_GT(condition_delta, 0.0F);
}

TEST(DiffusionHead, DdimAndDdpmAreDeterministicDistinctAndUnnormalized)
{
  DiffusionFixture fixture;
  fe::DiffusionHead head{fixture.tensors, fixture.arena};
  ASSERT_TRUE(head.valid());
  const std::vector<float> condition = seq(3uz, 0.31F, -0.2F);
  const std::vector<float> noise = seq(8uz, 0.29F, 0.1F);
  std::vector<float> workspace(head.sampler_workspace_size());
  std::vector<float> ddim_a(8uz), ddim_b(8uz), ddpm_a(8uz), ddpm_b(8uz), ddpm_other(8uz);
  ASSERT_TRUE(head.sample(condition, noise, 5uz, fe::DiffusionHead::kDDIM, 1u, workspace, ddim_a));
  ASSERT_TRUE(
      head.sample(condition, noise, 5uz, fe::DiffusionHead::kDDIM, 999u, workspace, ddim_b));
  ASSERT_TRUE(head.sample(condition, noise, 5uz, fe::DiffusionHead::kDDPM, 42u, workspace, ddpm_a));
  ASSERT_TRUE(head.sample(condition, noise, 5uz, fe::DiffusionHead::kDDPM, 42u, workspace, ddpm_b));
  ASSERT_TRUE(
      head.sample(condition, noise, 5uz, fe::DiffusionHead::kDDPM, 43u, workspace, ddpm_other));
  float scheduler_delta{0.0F};
  float seed_delta{0.0F};
  for (std::size_t i{0uz}; i < ddim_a.size(); ++i) {
    EXPECT_FLOAT_EQ(ddim_a[i], ddim_b[i]);
    EXPECT_FLOAT_EQ(ddpm_a[i], ddpm_b[i]);
    EXPECT_TRUE(std::isfinite(ddim_a[i]));
    EXPECT_TRUE(std::isfinite(ddpm_a[i]));
    const bool first_channel = i % 2uz == 0uz;
    EXPECT_GE(ddim_a[i], first_channel ? -2.0F : 0.0F);
    EXPECT_LE(ddim_a[i], first_channel ? 2.0F : 10.0F);
    scheduler_delta += std::fabs(ddim_a[i] - ddpm_a[i]);
    seed_delta += std::fabs(ddpm_a[i] - ddpm_other[i]);
  }
  EXPECT_GT(scheduler_delta, 0.0F);
  EXPECT_GT(seed_delta, 0.0F);
}

TEST(DiffusionHead, RejectsMalformedMetadata)
{
  DiffusionFixture fixture;
  fixture.storage[0][0] = 99.0F;
  fe::DiffusionHead head{fixture.tensors, fixture.arena};
  EXPECT_FALSE(head.valid());
}

TEST(DiffusionHead, RejectsTensorShapesThatExceedBackingStorage)
{
  DiffusionFixture metadata_fixture;
  metadata_fixture.tensors[0].bytes = sizeof(float);
  EXPECT_EQ(fe::DiffusionHead::required_workspace_floats(metadata_fixture.tensors), 0uz);
  fe::DiffusionHead metadata_head{metadata_fixture.tensors, metadata_fixture.arena};
  EXPECT_FALSE(metadata_head.valid());

  DiffusionFixture weight_fixture;
  const auto weight =
      std::ranges::find_if(weight_fixture.tensors, [](const fe::TensorView& tensor) {
        return tensor.name_view() == "dp.d0.r0.c1.w";
      });
  ASSERT_NE(weight, weight_fixture.tensors.end());
  weight->bytes = sizeof(float);
  fe::DiffusionHead weight_head{weight_fixture.tensors, weight_fixture.arena};
  EXPECT_FALSE(weight_head.valid());
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

TEST(FlowHead, GenerationCancellationStopsBetweenStepsAndReplaysDeterministically)
{
  FlowFixture fx;
  fe::FlowHead head{fx.views(), fx.arena};
  ASSERT_TRUE(head.valid());
  const std::vector<float> condition = seq(FlowFixture::kC, 0.1F, 0.0F);
  const std::vector<float> noise = seq(FlowFixture::kA, 0.3F, 0.2F);
  std::vector<float> workspace(head.sampler_workspace_size());
  std::vector<float> replay_workspace(head.sampler_workspace_size());
  std::vector<float> before_cancel(FlowFixture::kA), cancelled(FlowFixture::kA),
      replayed(FlowFixture::kA);
  std::atomic<std::uint64_t> newest_generation{7u};

  fe::FlowHead::SamplerState state{};
  ASSERT_TRUE(head.sampler_begin(condition, noise, 10uz, fe::FlowHead::kHeun, workspace, state,
                                 &newest_generation, 7u));
  EXPECT_EQ(head.sampler_advance(state, 2uz, before_cancel), 2uz);
  newest_generation.store(8u, std::memory_order_release);
  EXPECT_EQ(head.sampler_advance(state, 8uz, cancelled), 0uz);
  EXPECT_TRUE(state.cancelled);
  EXPECT_EQ(state.remaining(), 8uz);
  EXPECT_EQ(cancelled, before_cancel);

  fe::FlowHead::SamplerState replay{};
  ASSERT_TRUE(head.sampler_begin(condition, noise, 10uz, fe::FlowHead::kHeun, replay_workspace,
                                 replay, &newest_generation, 8u));
  EXPECT_EQ(head.sampler_advance(replay, 2uz, replayed), 2uz);
  EXPECT_EQ(replayed, before_cancel);
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
