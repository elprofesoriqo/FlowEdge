#pragma once

#include <charconv>
#include <cmath>
#include <string_view>
#include <type_traits>

namespace fe::relay::cli {

template<typename Number>
[[nodiscard]] bool parse_number(std::string_view text, Number& value) noexcept
{
  Number parsed{};
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), parsed);
  if (error != std::errc{} || end != text.data() + text.size())
    return false;
  if constexpr (std::is_floating_point_v<Number>)
    if (!std::isfinite(parsed))
      return false;
  value = parsed;
  return true;
}

template<typename Integer>
[[nodiscard]] bool parse_integer(std::string_view text, Integer& value) noexcept
{
  return parse_number(text, value);
}

} // namespace fe::relay::cli
