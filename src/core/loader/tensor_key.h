#pragma once

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <span>
#include <string_view>
#include <system_error>

namespace fe {

class TensorKeyBuilder
{
public:
  explicit TensorKeyBuilder(std::span<char> buf) noexcept : buf_{buf} {}

  bool append(std::string_view text) noexcept
  {
    if (text.size() > buf_.size() - pos_)
      return false;
    std::ranges::copy(text, buf_.begin() + static_cast<std::ptrdiff_t>(pos_));
    pos_ += text.size();
    return true;
  }

  bool append(std::size_t value) noexcept
  {
    if (pos_ >= buf_.size())
      return false;
    char* const first = buf_.data() + pos_;
    char* const last = buf_.data() + buf_.size();
    const auto [ptr, ec] = std::to_chars(first, last, value);
    if (ec != std::errc{})
      return false;
    pos_ += static_cast<std::size_t>(ptr - first);
    return true;
  }

  std::string_view view() noexcept
  {
    if (pos_ < buf_.size())
      buf_[pos_] = '\0';
    return {buf_.data(), pos_};
  }

private:
  std::span<char> buf_;
  std::size_t pos_{0uz};
};

} // namespace fe
