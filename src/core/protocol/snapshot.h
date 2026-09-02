#pragma once

#include "protocol/model_identity.h"

#include <cstddef>
#include <expected>
#include <limits>
#include <span>

namespace fe {

inline constexpr std::size_t kDecodeSnapshotHeaderBytes = 96uz;

struct SnapshotError
{
  int code{};
  const char* message{};
};

[[nodiscard]] constexpr std::size_t decode_snapshot_bytes(std::size_t payload_bytes) noexcept
{
  return payload_bytes <= (std::numeric_limits<std::size_t>::max() - kDecodeSnapshotHeaderBytes)
             ? kDecodeSnapshotHeaderBytes + payload_bytes
             : 0uz;
}

[[nodiscard]] std::expected<void, SnapshotError> export_decode_snapshot(
    const ModelIdentity& identity, std::span<const std::byte> payload,
    std::span<std::byte> destination) noexcept;

[[nodiscard]] std::expected<void, SnapshotError> import_decode_snapshot(
    const ModelIdentity& expected_identity, std::span<const std::byte> source,
    std::span<std::byte> payload) noexcept;

} // namespace fe
