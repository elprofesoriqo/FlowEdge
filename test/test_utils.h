#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace fe::test {

inline std::vector<float> seq(std::size_t size, float frequency, float phase)
{
  std::vector<float> values(size);
  for (std::size_t index{}; index < size; ++index)
    values[index] = std::sin(static_cast<float>(index) * frequency + phase);
  return values;
}

inline std::vector<std::uint16_t> to_bf16(const std::vector<float>& values)
{
  std::vector<std::uint16_t> result(values.size());
  for (std::size_t index{}; index < values.size(); ++index) {
    std::uint32_t bits{};
    std::memcpy(&bits, &values[index], sizeof(bits));
    result[index] = static_cast<std::uint16_t>(bits >> 16);
  }
  return result;
}

inline constexpr float k_tol{2e-3F};

} // namespace fe::test
