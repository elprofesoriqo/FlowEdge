#pragma once

#include "arena/thread_pool.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

// Mamba CPU kernels. Buffers are caller-owned and 64B-aligned (arena).
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
                                        std::size_t d_state, bool reset_state) noexcept;

FE_FORCE_ALIGN void matmul(std::span<const float> in, std::span<const float> w,
                           std::span<float> out, std::size_t rows, std::size_t in_dim,
                           std::size_t out_dim, ThreadPool* pool = nullptr) noexcept;
FE_FORCE_ALIGN void matmul(std::span<const float> in, std::span<const uint16_t> w,
                           std::span<float> out, std::size_t rows, std::size_t in_dim,
                           std::size_t out_dim, ThreadPool* pool = nullptr) noexcept;
} // namespace fe
