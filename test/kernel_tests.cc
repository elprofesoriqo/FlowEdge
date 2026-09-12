#include "arena/arena.h"
#include "arena/thread_pool.h"
#include "heads/diffusion/diffusion.h"
#include "heads/flow/flow.h"
#include "kernels/kernels.h"
#include "loader/safetensors.h"
#include "models/mamba/mamba.h"
#include "protocol/model_identity.h"
#include "protocol/snapshot.h"
#include "test/test_utils.h"

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

using fe::test::k_tol;
using fe::test::seq;

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
        EXPECT_NEAR(got[(r * out) + o], acc, k_tol) << "rows=" << rows;
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
    EXPECT_NEAR(x[i], in[i] / (1.0F + std::exp(-in[i])), k_tol);
}

TEST(Softplus, MatchesReference)
{
  std::vector<float> x = seq(37uz, 0.25F, 0.5F);
  const std::vector<float> in = x;
  fe::softplus(x);
  for (std::size_t i{0uz}; i < x.size(); ++i)
    EXPECT_NEAR(x[i], std::log1p(std::exp(in[i])), k_tol);
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
      EXPECT_NEAR(out[(r * dim) + i], in[(r * dim) + i] * scale, k_tol);
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
      EXPECT_NEAR(y[(c * len) + t], acc, k_tol);
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
    EXPECT_NEAR(y[i], yr[i], k_tol);
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

TEST(DiscretizeAndScan, SupportsStridedInputRows)
{
  constexpr std::size_t length{2uz};
  constexpr std::size_t d_inner{3uz};
  constexpr std::size_t d_state{2uz};
  constexpr std::size_t row_stride{5uz};
  const std::vector<float> delta{0.2F, 0.3F, 0.4F, 0.5F, 0.6F, 0.7F};
  const std::vector<float> a_neg{-1.0F, -2.0F, -3.0F, -0.5F, -1.5F, -2.5F};
  const std::vector<float> b{0.7F, -0.2F, 9.0F, 9.0F, 9.0F, 0.1F, 0.4F, 8.0F, 8.0F, 8.0F};
  const std::vector<float> u{0.4F, -0.6F, 0.8F, 0.2F, 0.3F, -0.1F};
  const std::vector<float> cp{0.9F, 0.25F, 7.0F, 7.0F, 7.0F, 0.5F, -0.3F, 6.0F, 6.0F, 6.0F};
  const std::vector<float> d{0.1F, 0.2F, 0.3F};
  std::vector<float> h(d_state * d_inner);
  std::vector<float> y(length * d_inner);

  fe::discretize_and_scan(delta, a_neg, b, u, cp, d, h, y, length, d_inner, d_state, true,
                          row_stride);

  std::vector<float> expected_state(d_state * d_inner);
  std::vector<float> expected(length * d_inner);
  for (std::size_t t{0uz}; t < length; ++t) {
    for (std::size_t c{0uz}; c < d_inner; ++c)
      expected[(t * d_inner) + c] = d[c] * u[(t * d_inner) + c];
    for (std::size_t n{0uz}; n < d_state; ++n)
      for (std::size_t c{0uz}; c < d_inner; ++c) {
        const float decay = std::exp(delta[(t * d_inner) + c] * a_neg[(n * d_inner) + c]);
        expected_state[(n * d_inner) + c] =
            decay * expected_state[(n * d_inner) + c] +
            delta[(t * d_inner) + c] * b[(t * row_stride) + n] * u[(t * d_inner) + c];
        expected[(t * d_inner) + c] += expected_state[(n * d_inner) + c] * cp[(t * row_stride) + n];
      }
  }
  for (std::size_t i{0uz}; i < y.size(); ++i)
    EXPECT_NEAR(y[i], expected[i], k_tol);
}
