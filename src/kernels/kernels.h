#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

// Mamba CPU kernels. Buffers are caller-owned and 64B-aligned (arena).
namespace fe {

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

// fused discretize and scan
FE_FORCE_ALIGN void discretize_and_scan(std::span<const float> delta, std::span<const float> a_log,
                                        std::span<const float> b, std::span<const float> u,
                                        std::span<const float> c_proj,
                                        std::span<const float> d_skip, std::span<float> h,
                                        std::span<float> y, std::span<float> a_work,
                                        std::size_t length, std::size_t d_inner,
                                        std::size_t d_state) noexcept;

FE_FORCE_ALIGN void matmul(std::span<const float> in, std::span<const float> w,
                           std::span<float> out, std::size_t rows, std::size_t in_dim,
                           std::size_t out_dim, class ThreadPool* pool = nullptr) noexcept;
FE_FORCE_ALIGN void matmul(std::span<const float> in, std::span<const uint16_t> w,
                           std::span<float> out, std::size_t rows, std::size_t in_dim,
                           std::size_t out_dim, class ThreadPool* pool = nullptr) noexcept;
} // namespace fe