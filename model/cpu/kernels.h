#pragma once

#include <cstddef>
#include <span>

// Mamba CPU kernels. Buffers are caller-owned and 64B-aligned (arena).

namespace fe {

// Depthwise causal conv1d, layout [channels][length]. x and y must not overlap.
// y[c,t] = bias[c] + Σ_k weight[c,k]·x[c, t-(kernel-1)+k]   (reads before x[0] are 0)
void conv1d_causal(std::span<const float> x, std::span<const float> weight,
                   std::span<const float> bias, std::span<float> y, std::size_t channels,
                   std::size_t length, std::size_t kernel) noexcept;

// SiLU gate: out = a · silu(g), silu(v) = v/(1+e^-v). May run in place.
void gate_silu(std::span<const float> a, std::span<const float> g, std::span<float> out) noexcept;

// Selective scan over the discretized parameters Ā (delta_a) and B̄u (delta_bu),
// state-major layout [t][n][c] so the channel loop stays contiguous:
//   h[n,c] = delta_a·h[n,c] + delta_bu
//   y[t,c] = d_skip·u + Σ_n h[n,c]·c_proj[t,n]
// h is [d_state][d_inner] scratch, zeroed here at t=0. Buffers must be distinct.
void selective_scan(std::span<const float> delta_a, std::span<const float> delta_bu,
                    std::span<const float> c_proj, std::span<const float> d_skip,
                    std::span<const float> u, std::span<float> h, std::span<float> y,
                    std::size_t length, std::size_t d_inner, std::size_t d_state) noexcept;

} // namespace fe
