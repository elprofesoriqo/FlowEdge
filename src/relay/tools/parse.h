#pragma once

#include <charconv>
#include <string_view>

namespace fe::relay::cli {

template<typename Number>
[[nodiscard]] bool parse_number(std::string_view text, Number& value) noexcept
{
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc{} && end == text.data() + text.size();
}

template<typename Integer>
[[nodiscard]] bool parse_integer(std::string_view text, Integer& value) noexcept
{
  return parse_number(text, value);
}

} // namespace fe::relay::cli
