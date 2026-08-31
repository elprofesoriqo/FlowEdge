#pragma once

#include "relay/jobs/job_types.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <span>
#include <string_view>

namespace fe::relay {

inline constexpr std::size_t kStateCapsuleHeaderBytes = 96uz;
inline constexpr std::size_t kStateCapsuleChecksumBytes = 16uz;

enum class StateCapsuleErrorCode : std::uint8_t
{
  kBufferTooSmall,
  kTruncated,
  kInvalidHeader,
  kUnsupportedVersion,
  kInvalidMetadata,
  kChecksumMismatch,
};

struct StateCapsuleError
{
  StateCapsuleErrorCode code{};
  std::string_view message{};
};

struct StateCapsuleMetadata
{
  JobDescriptor descriptor{};
  JobState state{JobState::kReady};
  std::uint64_t completed_work_units{};
};

struct StateCapsuleView
{
  StateCapsuleMetadata metadata{};
  std::span<const std::byte> payload{};
};

[[nodiscard]] constexpr std::size_t state_capsule_bytes(std::size_t payload_bytes) noexcept
{
  constexpr std::size_t overhead = kStateCapsuleHeaderBytes + kStateCapsuleChecksumBytes;
  return payload_bytes <= std::numeric_limits<std::size_t>::max() - overhead
             ? overhead + payload_bytes
             : 0uz;
}

[[nodiscard]] std::expected<std::size_t, StateCapsuleError> write_state_capsule(
    const StateCapsuleMetadata& metadata, std::span<const std::byte> payload,
    std::span<std::byte> destination) noexcept;

[[nodiscard]] std::expected<StateCapsuleView, StateCapsuleError> read_state_capsule(
    std::span<const std::byte> source) noexcept;

} // namespace fe::relay
