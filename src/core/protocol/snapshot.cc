#include "protocol/snapshot.h"

#include "protocol/byte_codec.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <ranges>

namespace fe {
namespace {

constexpr std::array<std::byte, 8> kMagic{std::byte{'F'}, std::byte{'E'}, std::byte{'S'},
                                          std::byte{'N'}, std::byte{'A'}, std::byte{'P'},
                                          std::byte{'0'}, std::byte{'1'}};
constexpr std::uint16_t kVersion = 1u;
using protocol::read_le;
using protocol::write_le;

[[nodiscard]] std::uint64_t payload_checksum(std::span<const std::byte> payload) noexcept
{
  const ModelDigest digest = fingerprint_bytes(payload);
  std::uint64_t value{0u};
  for (std::size_t i{0uz}; i < sizeof(value); ++i)
    value |= static_cast<std::uint64_t>(digest[i]) << (8uz * i);
  return value;
}

} // namespace

std::expected<void, SnapshotError> export_decode_snapshot(const ModelIdentity& identity,
                                                          std::span<const std::byte> payload,
                                                          std::span<std::byte> destination) noexcept
{
  const std::size_t required = decode_snapshot_bytes(payload.size());
  if (required == 0uz || destination.size() < required)
    return std::unexpected(
        SnapshotError{.code = 1, .message = "Decode snapshot destination is too small"});

  std::ranges::fill(destination.first(required), std::byte{0});
  std::ranges::copy(kMagic, destination.begin());
  write_le(destination, 8uz, kVersion);
  write_le(destination, 10uz, static_cast<std::uint16_t>(kDecodeSnapshotHeaderBytes));
  write_le(destination, 12uz, std::uint32_t{0u});
  write_le(destination, 16uz, identity.architecture);
  write_le(destination, 20uz, identity.precision);
  for (std::size_t i{0uz}; i < identity.digest.size(); ++i)
    destination[24uz + i] = static_cast<std::byte>(identity.digest[i]);
  write_le(destination, 40uz, identity.d_model);
  write_le(destination, 48uz, identity.n_layers);
  write_le(destination, 56uz, identity.d_inner);
  write_le(destination, 64uz, identity.d_state);
  write_le(destination, 72uz, identity.d_conv);
  write_le(destination, 80uz, static_cast<std::uint64_t>(payload.size()));
  write_le(destination, 88uz, payload_checksum(payload));
  std::ranges::copy(payload, destination.begin() + kDecodeSnapshotHeaderBytes);
  return {};
}

std::expected<void, SnapshotError> import_decode_snapshot(const ModelIdentity& expected,
                                                          std::span<const std::byte> source,
                                                          std::span<std::byte> payload) noexcept
{
  if (source.size() < kDecodeSnapshotHeaderBytes)
    return std::unexpected(SnapshotError{.code = 1, .message = "Decode snapshot is truncated"});
  if (!std::ranges::equal(kMagic, source.first(kMagic.size())))
    return std::unexpected(
        SnapshotError{.code = 10, .message = "Decode snapshot magic is invalid"});
  if (read_le<std::uint16_t>(source, 8uz) != kVersion ||
      read_le<std::uint16_t>(source, 10uz) != kDecodeSnapshotHeaderBytes)
    return std::unexpected(
        SnapshotError{.code = 9, .message = "Decode snapshot version is unsupported"});
  if (read_le<std::uint32_t>(source, 12uz) != 0u)
    return std::unexpected(
        SnapshotError{.code = 10, .message = "Decode snapshot header flags are invalid"});

  const auto state_bytes = read_le<std::uint64_t>(source, 80uz);
  if (state_bytes != payload.size() || state_bytes > source.size() - kDecodeSnapshotHeaderBytes ||
      source.size() != kDecodeSnapshotHeaderBytes + state_bytes)
    return std::unexpected(
        SnapshotError{.code = 1, .message = "Decode snapshot size is incompatible or truncated"});

  ModelDigest digest{};
  for (std::size_t i{0uz}; i < digest.size(); ++i)
    digest[i] = std::to_integer<std::uint8_t>(source[24uz + i]);
  const bool compatible = read_le<std::uint32_t>(source, 16uz) == expected.architecture &&
                          read_le<std::uint32_t>(source, 20uz) == expected.precision &&
                          digest == expected.digest &&
                          read_le<std::uint64_t>(source, 40uz) == expected.d_model &&
                          read_le<std::uint64_t>(source, 48uz) == expected.n_layers &&
                          read_le<std::uint64_t>(source, 56uz) == expected.d_inner &&
                          read_le<std::uint64_t>(source, 64uz) == expected.d_state &&
                          read_le<std::uint64_t>(source, 72uz) == expected.d_conv;
  if (!compatible)
    return std::unexpected(
        SnapshotError{.code = 9, .message = "Decode snapshot belongs to an incompatible model"});

  const auto encoded_payload = source.subspan(kDecodeSnapshotHeaderBytes);
  if (read_le<std::uint64_t>(source, 88uz) != payload_checksum(encoded_payload))
    return std::unexpected(
        SnapshotError{.code = 10, .message = "Decode snapshot checksum mismatch"});
  std::ranges::copy(encoded_payload, payload.begin());
  return {};
}

} // namespace fe
