#pragma once

#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>

namespace fe::protocol {

template<std::integral Integer>
void write_le(std::span<std::byte> destination, std::size_t offset, Integer value) noexcept
{
  using Unsigned = std::make_unsigned_t<Integer>;
  const Unsigned bits = static_cast<Unsigned>(value);
  for (std::size_t index{}; index < sizeof(Integer); ++index)
    destination[offset + index] = static_cast<std::byte>((bits >> (index * 8uz)) & Unsigned{0xffu});
}

template<std::integral Integer>
[[nodiscard]] Integer read_le(std::span<const std::byte> source, std::size_t offset) noexcept
{
  using Unsigned = std::make_unsigned_t<Integer>;
  Unsigned value{};
  for (std::size_t index{}; index < sizeof(Integer); ++index)
    value |= static_cast<Unsigned>(std::to_integer<unsigned char>(source[offset + index]))
             << (index * 8uz);
  return static_cast<Integer>(value);
}

} // namespace fe::protocol
