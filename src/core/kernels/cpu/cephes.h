#pragma once

#include <numbers>

// Cephes single-precision exp/log polynomial constants, shared by the SIMD kernels (avx2/neon).
namespace fe {

inline constexpr float exp_clamp = 88.3762F;
inline constexpr float log2e = std::numbers::log2e_v<float>;
inline constexpr float ln2_hi = 0.693359375F;
inline constexpr float ln2_lo = -2.12194440e-4F;
inline constexpr float exp_p0 = 1.9875691500e-4F;
inline constexpr float exp_p1 = 1.3981999507e-3F;
inline constexpr float exp_p2 = 8.3334519073e-3F;
inline constexpr float exp_p3 = 4.1665795894e-2F;
inline constexpr float exp_p4 = 1.6666665459e-1F;
inline constexpr float exp_p5 = 5.0000001201e-1F;
inline constexpr float sqrt_half = 0.707106781F; // Cephes logf mantissa split point
inline constexpr float log_p0 = 7.0376836292e-2F;
inline constexpr float log_p1 = -1.1514610310e-1F;
inline constexpr float log_p2 = 1.1676998740e-1F;
inline constexpr float log_p3 = -1.2420140846e-1F;
inline constexpr float log_p4 = 1.4249322787e-1F;
inline constexpr float log_p5 = -1.6668057665e-1F;
inline constexpr float log_p6 = 2.0000714765e-1F;
inline constexpr float log_p7 = -2.4999993993e-1F;
inline constexpr float log_p8 = 3.3333331174e-1F;

} // namespace fe
