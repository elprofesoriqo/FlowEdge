#pragma once

#include <cstddef>
#include <cstdint>

// C++17 device-op surface used by the C++23 kernels.h bridge. Host pointers.
// Buffers on device are grow-only; the first call of a given size may allocate.
namespace fe::cuda_ops {

[[nodiscard]] bool device_available() noexcept;

void conv1d_causal(const float* x, const float* weight, const float* bias, float* y,
                   std::size_t channels, std::size_t length, std::size_t kernel) noexcept;
void conv1d_step(const float* window, const float* weight, const float* bias, float* y,
                 std::size_t channels, std::size_t kernel) noexcept;
void rmsnorm(const float* in, const float* weight, float* out, std::size_t rows,
             std::size_t dim) noexcept;
void gate_silu(const float* a, const float* g, float* out, std::size_t n) noexcept;
void silu(float* x, std::size_t n) noexcept;
void softplus(float* x, std::size_t n) noexcept;
void matmul_f32(const float* in, const float* w, float* out, std::size_t rows, std::size_t in_dim,
                std::size_t out_dim) noexcept;
void matmul_bf16(const float* in, const std::uint16_t* w, float* out, std::size_t rows,
                 std::size_t in_dim, std::size_t out_dim) noexcept;
void discretize_and_scan(const float* delta, const float* a_neg, const float* b, const float* u,
                         const float* c_proj, const float* d_skip, float* h, float* y,
                         std::size_t length, std::size_t d_inner, std::size_t d_state,
                         bool reset_state, std::size_t row_stride) noexcept;

} // namespace fe::cuda_ops
