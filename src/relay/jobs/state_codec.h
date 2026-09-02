#pragma once

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <span>
#include <type_traits>

namespace fe::relay {

enum class StateCodecError : std::uint8_t
{
  kTruncated,
};

class StateWriter
{
public:
  explicit StateWriter(std::span<std::byte> destination) noexcept : destination_{destination} {}

  template<std::integral Integer>
    requires(!std::same_as<std::remove_cv_t<Integer>, bool>)
  [[nodiscard]] bool write(Integer value) noexcept
  {
    if (remaining() < sizeof(Integer))
      return false;
    using Unsigned = std::make_unsigned_t<Integer>;
    const auto bits = static_cast<Unsigned>(value);
    for (std::size_t index{}; index < sizeof(Integer); ++index) {
      destination_[position_ + index] = static_cast<std::byte>(
          (bits >> (index * 8uz)) & Unsigned{0xffu}); // NOLINT(bugprone-signed-bitwise)
    }
    position_ += sizeof(Integer);
    return true;
  }

  [[nodiscard]] bool write_float(float value) noexcept
  {
    return write(std::bit_cast<std::uint32_t>(value));
  }

  [[nodiscard]] bool write_bytes(std::span<const std::byte> bytes) noexcept
  {
    if (bytes.size() > remaining())
      return false;
    for (const std::byte byte : bytes)
      destination_[position_++] = byte;
    return true;
  }

  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  [[nodiscard]] std::size_t remaining() const noexcept { return destination_.size() - position_; }

private:
  std::span<std::byte> destination_{};
  std::size_t position_{};
};

class StateReader
{
public:
  explicit StateReader(std::span<const std::byte> source) noexcept : source_{source} {}

  template<std::integral Integer>
    requires(!std::same_as<std::remove_cv_t<Integer>, bool>)
  [[nodiscard]] std::expected<Integer, StateCodecError> read() noexcept
  {
    if (remaining() < sizeof(Integer))
      return std::unexpected(StateCodecError::kTruncated);
    using Unsigned = std::make_unsigned_t<Integer>;
    Unsigned value{};
    for (std::size_t index{}; index < sizeof(Integer); ++index) {
      // NOLINTNEXTLINE(bugprone-signed-bitwise): accumulation uses the unsigned representation.
      value |= static_cast<Unsigned>(std::to_integer<std::uint8_t>(source_[position_ + index]))
               << (index * 8uz);
    }
    position_ += sizeof(Integer);
    return static_cast<Integer>(value);
  }

  [[nodiscard]] std::expected<float, StateCodecError> read_float() noexcept
  {
    const auto bits = read<std::uint32_t>();
    if (!bits)
      return std::unexpected(bits.error());
    return std::bit_cast<float>(*bits);
  }

  [[nodiscard]] std::expected<std::span<const std::byte>, StateCodecError> read_bytes(
      std::size_t bytes) noexcept
  {
    if (bytes > remaining())
      return std::unexpected(StateCodecError::kTruncated);
    const auto result = source_.subspan(position_, bytes);
    position_ += bytes;
    return result;
  }

  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  [[nodiscard]] std::size_t remaining() const noexcept { return source_.size() - position_; }

private:
  std::span<const std::byte> source_{};
  std::size_t position_{};
};

} // namespace fe::relay
