#pragma once

#include <cstddef>
#include <span>

namespace fe {

inline void copy_span(std::span<const float> src, std::span<float> dst) noexcept
{
  for (std::size_t i{0uz}; i < src.size(); ++i)
    dst[i] = src[i];
}

inline void add_inplace(std::span<float> x, std::span<const float> y) noexcept
{
  for (std::size_t i{0uz}; i < x.size(); ++i)
    x[i] += y[i];
}

inline void add3_inplace(std::span<float> x, std::span<const float> y,
                         std::span<const float> z) noexcept
{
  for (std::size_t i{0uz}; i < x.size(); ++i)
    x[i] += y[i] + z[i];
}

inline void add_scaled(std::span<float> x, std::span<const float> dx, float scale) noexcept
{
  for (std::size_t i{0uz}; i < x.size(); ++i)
    x[i] += scale * dx[i];
}

inline void scaled_sum(std::span<const float> x, std::span<const float> dx, float scale,
                       std::span<float> out) noexcept
{
  for (std::size_t i{0uz}; i < x.size(); ++i)
    out[i] = x[i] + (scale * dx[i]);
}

inline void add_heun(std::span<float> x, std::span<const float> k1, std::span<const float> k2,
                     float dt) noexcept
{
  for (std::size_t i{0uz}; i < x.size(); ++i)
    x[i] += 0.5F * dt * (k1[i] + k2[i]);
}

inline void add_rk4(std::span<float> x, std::span<const float> k1, std::span<const float> k2,
                    std::span<const float> k3, std::span<const float> k4, float dt) noexcept
{
  for (std::size_t i{0uz}; i < x.size(); ++i)
    x[i] += (dt / 6.0F) * (k1[i] + (2.0F * k2[i]) + (2.0F * k3[i]) + k4[i]);
}

} // namespace fe
