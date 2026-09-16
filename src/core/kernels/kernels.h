#pragma once

#include "arena/thread_pool.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

// CPU kernels. Buffers are caller-owned and 64B-aligned (arena).
namespace fe {

enum class MatmulWeightType : std::uint8_t
{
  kF32,
  kBF16,
};

// Choose a power-of-two task count from the available pool capacity. This keeps
// dispatch overhead out of small/cache-resident projections and avoids creating
// more output slices than can do useful work.
[[nodiscard]] constexpr unsigned matmul_task_count(unsigned available, std::size_t rows,
                                                   std::size_t in_dim, std::size_t out_dim,
                                                   MatmulWeightType weight_type) noexcept
{
  if (available < 2u || rows == 0uz || in_dim == 0uz || out_dim < 128uz)
    return 1u;

  constexpr std::size_t kMinOutputsPerTask = 64uz;
  constexpr std::size_t kF32WorkPerTask = 786'432uz;
  constexpr std::size_t kBF16WorkPerTask = 524'288uz;
  constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
  const auto saturating_mul = [](std::size_t lhs, std::size_t rhs) constexpr noexcept {
    return (lhs != 0uz && rhs > (kMax / lhs)) ? kMax : lhs * rhs;
  };
  const std::size_t work = saturating_mul(saturating_mul(rows, in_dim), out_dim);
  const std::size_t work_per_task =
      (weight_type == MatmulWeightType::kBF16) ? kBF16WorkPerTask : kF32WorkPerTask;
  const std::size_t work_limited = 1uz + ((work - 1uz) / work_per_task);
  const std::size_t output_limited = out_dim / kMinOutputsPerTask;
  const auto useful =
      static_cast<unsigned>(std::min<std::size_t>({available, work_limited, output_limited}));
  return useful >= 2u ? std::bit_floor(useful) : 1u;
}

[[nodiscard]] inline unsigned matmul_task_count(const ThreadPool* pool, std::size_t rows,
                                                std::size_t in_dim, std::size_t out_dim,
                                                MatmulWeightType weight_type) noexcept
{
  return matmul_task_count(pool != nullptr ? pool->nthreads() : 0u, rows, in_dim, out_dim,
                           weight_type);
}

#if defined(_WIN32) && defined(__GNUC__)
#define FE_FORCE_ALIGN __attribute__((force_align_arg_pointer))
#else
#define FE_FORCE_ALIGN
#endif

// layout [channels][length]
// y[c,t] = bias[c] + Σ_k weight[c,k]·x[c, t-(kernel-1)+k]
FE_FORCE_ALIGN void conv1d_causal(std::span<const float> x, std::span<const float> weight,
                                  std::span<const float> bias, std::span<float> y,
                                  std::size_t channels, std::size_t length,
                                  std::size_t kernel) noexcept;

// Workspace for packing Conv1D through matmul: [L_out][IC*K] plus [L_out][OC].
[[nodiscard]] constexpr std::size_t conv1d_workspace_floats(std::size_t in_channels,
                                                            std::size_t out_channels,
                                                            std::size_t output_length,
                                                            std::size_t kernel) noexcept
{
  return (output_length * in_channels * kernel) + (output_length * out_channels);
}

// Workspace for packing ConvTranspose1D: X^T [L_in][IC], W_k [OC][IC], GEMM [L_in][OC].
[[nodiscard]] constexpr std::size_t conv_transpose1d_workspace_floats(
    std::size_t in_channels, std::size_t out_channels, std::size_t input_length) noexcept
{
  return (input_length * in_channels) + (out_channels * in_channels) +
         (input_length * out_channels);
}

// Dense PyTorch-style 1-D cross-correlation. Activations use [channels][length]
// and weights use [out_channels][in_channels][kernel]. When `workspace` is large
// enough, the kernel packs columns and reuses `matmul`; otherwise it keeps the
// direct loop. No heap allocation.
FE_FORCE_ALIGN void conv1d(std::span<const float> x, std::span<const float> weight,
                           std::span<const float> bias, std::span<float> y, std::size_t in_channels,
                           std::size_t out_channels, std::size_t input_length,
                           std::size_t output_length, std::size_t kernel, std::size_t stride,
                           std::size_t padding, ThreadPool* pool = nullptr,
                           std::span<float> workspace = {}) noexcept;

// PyTorch ConvTranspose1d layout: weights are [in_channels][out_channels][kernel].
FE_FORCE_ALIGN void conv_transpose1d(std::span<const float> x, std::span<const float> weight,
                                     std::span<const float> bias, std::span<float> y,
                                     std::size_t in_channels, std::size_t out_channels,
                                     std::size_t input_length, std::size_t output_length,
                                     std::size_t kernel, std::size_t stride, std::size_t padding,
                                     ThreadPool* pool = nullptr,
                                     std::span<float> workspace = {}) noexcept;

// GroupNorm over a single [channels][length] sample, with per-channel affine terms.
FE_FORCE_ALIGN void group_norm(std::span<float> x, std::span<const float> weight,
                               std::span<const float> bias, std::size_t channels,
                               std::size_t length, std::size_t groups,
                               float epsilon = 1e-5F) noexcept;

// In-place Mish: x * tanh(softplus(x)).
FE_FORCE_ALIGN void mish(std::span<float> x) noexcept;

// In-place FiLM used by LeRobot residual blocks: x[c,t] = scale[c]*x[c,t] + bias[c].
FE_FORCE_ALIGN void film(std::span<float> x, std::span<const float> scale,
                         std::span<const float> bias, std::size_t channels,
                         std::size_t length) noexcept;

// LeRobot DiffusionSinusoidalPosEmb. `out.size()` must be a positive even number.
FE_FORCE_ALIGN void diffusion_timestep_embedding(float timestep, std::span<float> out) noexcept;

// SiLU gate: out = a · silu(g), silu(v) = v/(1+e^-v)
FE_FORCE_ALIGN void gate_silu(std::span<const float> a, std::span<const float> g,
                              std::span<float> out) noexcept;

// In-place SiLU: x[i] = x[i]/(1+e^-x[i])
FE_FORCE_ALIGN void silu(std::span<float> x) noexcept;

// In-place softplus: x[i] = log1p(e^x[i])
FE_FORCE_ALIGN void softplus(std::span<float> x) noexcept;

// RMSNorm over rows of `dim`: out[r,i] = in[r,i] / sqrt(mean_i(in[r,:]²)+eps) · weight[i]
FE_FORCE_ALIGN void rmsnorm(std::span<const float> in, std::span<const float> weight,
                            std::span<float> out, std::size_t rows, std::size_t dim) noexcept;

// Affine LayerNorm over rows. Kept scalar and allocation-free so model backbones
// can use it on every supported CPU before architecture-specific tuning.
inline void layer_norm(std::span<const float> in, std::span<const float> weight,
                       std::span<const float> bias, std::span<float> out, std::size_t rows,
                       std::size_t dim, float epsilon = 1e-5F) noexcept
{
  for (std::size_t r{}; r < rows; ++r) {
    const float* const input = in.data() + (r * dim);
    float* const output = out.data() + (r * dim);
    float mean{};
    for (std::size_t i{}; i < dim; ++i)
      mean += input[i];
    mean /= static_cast<float>(dim);
    float variance{};
    for (std::size_t i{}; i < dim; ++i) {
      const float centered = input[i] - mean;
      variance += centered * centered;
    }
    const float scale = 1.0F / std::sqrt((variance / static_cast<float>(dim)) + epsilon);
    for (std::size_t i{}; i < dim; ++i)
      output[i] = ((input[i] - mean) * scale * weight[i]) + bias[i];
  }
}

// Exact-tanh GELU used by the initial Transformer contract.
inline void gelu(std::span<float> values) noexcept
{
  constexpr float kSqrtTwoOverPi = 0.7978845608028654F;
  constexpr float kCoefficient = 0.044715F;
  for (float& value : values) {
    const float cube = value * value * value;
    value = 0.5F * value * (1.0F + std::tanh(kSqrtTwoOverPi * (value + (kCoefficient * cube))));
  }
}

// Stable in-place softmax for one contiguous score row.
inline void softmax(std::span<float> values) noexcept
{
  if (values.empty())
    return;
  const float maximum = *std::max_element(values.begin(), values.end());
  float sum{};
  for (float& value : values) {
    value = std::exp(value - maximum);
    sum += value;
  }
  const float inverse = 1.0F / sum;
  for (float& value : values)
    value *= inverse;
}

// Round F32 values to BF16-representable F32 (round-to-nearest-even).
inline void round_to_bf16(std::span<float> values) noexcept
{
  for (float& value : values) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t bias = 0x7FFFu + ((bits >> 16u) & 1u);
    value = std::bit_cast<float>((bits + bias) & 0xFFFF0000u);
  }
}

// Llama-style RMSNorm with BF16 rounding after the reduction and after the
// learned weight. SmolVLA's expert depends on that rounding path; the generic
// `rmsnorm` kernel multiplies through in F32.
inline void rmsnorm_bf16(std::span<const float> input, std::span<const float> weight,
                         std::span<float> output, std::size_t rows, std::size_t width,
                         float epsilon = 1e-5F) noexcept
{
  for (std::size_t row{}; row < rows; ++row) {
    const float* const source = input.data() + (row * width);
    float* const destination = output.data() + (row * width);
    float squared_sum{};
    for (std::size_t channel{}; channel < width; ++channel)
      squared_sum += source[channel] * source[channel];
    const float scale = 1.0F / std::sqrt((squared_sum / static_cast<float>(width)) + epsilon);
    for (std::size_t channel{}; channel < width; ++channel)
      destination[channel] = source[channel] * scale;
    round_to_bf16({destination, width});
    for (std::size_t channel{}; channel < width; ++channel)
      destination[channel] *= weight[channel];
    round_to_bf16({destination, width});
  }
}

// Grouped-query attention for one expert layer. Query is [rows][query_heads][head_width].
// Cross-attention reads keys/values as the projected prefix. Self-attention reads the
// VLM cache from prefix_keys/prefix_values and the projected suffix from keys/values,
// with a causal suffix of length row+1. `scores` must hold prefix_size+rows floats.
// Softmax probabilities are rounded to BF16 to match the SmolVLA source dtypes.
inline void grouped_query_attention(
    std::span<const float> query, std::span<const float> keys, std::span<const float> values,
    std::span<const float> prefix_keys, std::span<const float> prefix_values,
    std::span<const std::uint8_t> prefix_mask, std::span<float> scores, std::span<float> attended,
    std::size_t rows, std::size_t query_heads, std::size_t kv_heads, std::size_t head_width,
    std::size_t prefix_size, bool cross_attention) noexcept
{
  if (query_heads == 0uz || kv_heads == 0uz || (query_heads % kv_heads) != 0uz || head_width == 0uz)
    return;
  const std::size_t query_floats = rows * query_heads * head_width;
  const std::size_t key_rows = cross_attention ? prefix_size : rows;
  const std::size_t kv_floats = key_rows * kv_heads * head_width;
  const std::size_t prefix_kv = prefix_size * kv_heads * head_width;
  const std::size_t score_need = cross_attention ? prefix_size : prefix_size + rows;
  if (query.size() < query_floats || keys.size() < kv_floats || values.size() < kv_floats ||
      prefix_mask.size() < prefix_size || scores.size() < score_need ||
      attended.size() < query_floats)
    return;
  if (!cross_attention && (prefix_keys.size() < prefix_kv || prefix_values.size() < prefix_kv))
    return;
  const std::size_t groups = query_heads / kv_heads;
  for (std::size_t row{}; row < rows; ++row) {
    for (std::size_t head{}; head < query_heads; ++head) {
      const std::size_t kv_head = head / groups;
      const float* const q = query.data() + ((row * query_heads + head) * head_width);
      const std::size_t attended_offset = (row * query_heads * head_width) + (head * head_width);
      const std::size_t score_count = cross_attention ? prefix_size : prefix_size + row + 1uz;
      for (std::size_t token{}; token < prefix_size; ++token) {
        if (prefix_mask[token] == 0u) {
          scores[token] = -std::numeric_limits<float>::infinity();
          continue;
        }
        const float* const key = (cross_attention ? keys.data() : prefix_keys.data()) +
                                 ((token * kv_heads + kv_head) * head_width);
        float dot{};
        for (std::size_t channel{}; channel < head_width; ++channel)
          dot += q[channel] * key[channel];
        scores[token] = dot / std::sqrt(static_cast<float>(head_width));
      }
      if (!cross_attention) {
        for (std::size_t token{}; token <= row; ++token) {
          const float* const key = keys.data() + ((token * kv_heads + kv_head) * head_width);
          float dot{};
          for (std::size_t channel{}; channel < head_width; ++channel)
            dot += q[channel] * key[channel];
          scores[prefix_size + token] = dot / std::sqrt(static_cast<float>(head_width));
        }
      }
      std::span<float> probabilities = scores.first(score_count);
      softmax(probabilities);
      round_to_bf16(probabilities);
      for (std::size_t channel{}; channel < head_width; ++channel) {
        float sum{};
        for (std::size_t token{}; token < prefix_size; ++token) {
          if (prefix_mask[token] == 0u)
            continue;
          const float* const value = (cross_attention ? values.data() : prefix_values.data()) +
                                     ((token * kv_heads + kv_head) * head_width);
          sum += probabilities[token] * value[channel];
        }
        if (!cross_attention) {
          for (std::size_t token{}; token <= row; ++token)
            sum += probabilities[prefix_size + token] *
                   values[(token * kv_heads * head_width) + (kv_head * head_width) + channel];
        }
        attended[attended_offset + channel] = sum;
      }
    }
  }
}

// Interleaved RoPE on [rows][heads][head_width] with even head_width.
inline void apply_rope(std::span<float> values, std::size_t rows, std::size_t heads,
                       std::size_t head_width, std::size_t position_base) noexcept
{
  if (head_width < 2uz || (head_width % 2uz) != 0uz)
    return;
  constexpr float kLogWavelength = 9.210340371976184F; // log(10,000)
  const std::size_t half = head_width / 2uz;
  for (std::size_t row{}; row < rows; ++row) {
    const float position = static_cast<float>(position_base + row);
    for (std::size_t head{}; head < heads; ++head) {
      float* const data = values.data() + ((row * heads + head) * head_width);
      for (std::size_t channel{}; channel < half; ++channel) {
        const float exponent =
            (2.0F * static_cast<float>(channel)) / static_cast<float>(head_width);
        const float angle = position / std::exp(kLogWavelength * exponent);
        const float sine = std::sin(angle);
        const float cosine = std::cos(angle);
        const float first = data[channel];
        const float second = data[half + channel];
        data[channel] = (first * cosine) - (second * sine);
        data[half + channel] = (second * cosine) + (first * sine);
      }
    }
  }
}

// 1 causal-conv step: window is [channels][kernel] with the newest sample at index kernel-1.
// y[c] = bias[c] + sum_k weight[c,k]·window[c,k]
FE_FORCE_ALIGN void conv1d_step(std::span<const float> window, std::span<const float> weight,
                                std::span<const float> bias, std::span<float> y,
                                std::size_t channels, std::size_t kernel) noexcept;

// Fused selective scan. `a_neg` is the load-time-transformed
// A[n,c] = -exp(A_log[c,n]); `reset_state` distinguishes a prefill from a
// streaming continuation.
FE_FORCE_ALIGN void discretize_and_scan(std::span<const float> delta, std::span<const float> a_neg,
                                        std::span<const float> b, std::span<const float> u,
                                        std::span<const float> c_proj,
                                        std::span<const float> d_skip, std::span<float> h,
                                        std::span<float> y, std::size_t length, std::size_t d_inner,
                                        std::size_t d_state, bool reset_state,
                                        std::size_t row_stride = 0uz) noexcept;

FE_FORCE_ALIGN void matmul(std::span<const float> in, std::span<const float> w,
                           std::span<float> out, std::size_t rows, std::size_t in_dim,
                           std::size_t out_dim, ThreadPool* pool = nullptr) noexcept;
FE_FORCE_ALIGN void matmul(std::span<const float> in, std::span<const uint16_t> w,
                           std::span<float> out, std::size_t rows, std::size_t in_dim,
                           std::size_t out_dim, ThreadPool* pool = nullptr) noexcept;
} // namespace fe
