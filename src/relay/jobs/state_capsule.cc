#include "relay/jobs/state_capsule.h"

#include "protocol/byte_codec.h"
#include "protocol/model_identity.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <ranges>

namespace fe::relay {
namespace {

constexpr std::array<std::byte, 8> kMagic{std::byte{'F'}, std::byte{'E'}, std::byte{'C'},
                                          std::byte{'A'}, std::byte{'P'}, std::byte{'0'},
                                          std::byte{'0'}, std::byte{'1'}};
constexpr std::uint16_t kVersion = 1u;
using fe::protocol::read_le;
using fe::protocol::write_le;

[[nodiscard]] bool valid_progress(const StateCapsuleMetadata& metadata) noexcept
{
  if (!valid_job_descriptor(metadata.descriptor) || !valid_job_state(metadata.state) ||
      metadata.completed_work_units > metadata.descriptor.total_work_units)
    return false;
  if (metadata.state == JobState::kReady)
    return metadata.completed_work_units == 0u;
  if (metadata.state == JobState::kRunning)
    return metadata.completed_work_units < metadata.descriptor.total_work_units;
  if (metadata.state == JobState::kComplete)
    return metadata.completed_work_units == metadata.descriptor.total_work_units;
  return true;
}

} // namespace

std::expected<std::size_t, StateCapsuleError> write_state_capsule(
    const StateCapsuleMetadata& metadata, std::span<const std::byte> payload,
    std::span<std::byte> destination) noexcept
{
  if (!valid_progress(metadata))
    return std::unexpected(StateCapsuleError{StateCapsuleErrorCode::kInvalidMetadata,
                                             "State capsule metadata is invalid"});
  const std::size_t required = state_capsule_bytes(payload.size());
  if (required == 0uz || destination.size() < required)
    return std::unexpected(StateCapsuleError{StateCapsuleErrorCode::kBufferTooSmall,
                                             "State capsule destination is too small"});

  std::ranges::fill(destination.first(kStateCapsuleHeaderBytes), std::byte{0});
  std::ranges::copy(kMagic, destination.begin());
  write_le(destination, 8uz, kVersion);
  write_le(destination, 10uz, static_cast<std::uint16_t>(kStateCapsuleHeaderBytes));
  write_le(destination, 12uz, static_cast<std::uint16_t>(metadata.descriptor.kind));
  write_le(destination, 14uz, static_cast<std::uint16_t>(metadata.state));
  write_le(destination, 16uz, std::uint32_t{0u});
  write_le(destination, 20uz, std::uint32_t{0u});
  for (std::size_t index{}; index < metadata.descriptor.model_digest.size(); ++index)
    destination[24uz + index] = static_cast<std::byte>(metadata.descriptor.model_digest[index]);
  write_le(destination, 40uz, metadata.descriptor.session_id);
  write_le(destination, 48uz, metadata.descriptor.generation);
  write_le(destination, 56uz, metadata.descriptor.deadline_ns);
  write_le(destination, 64uz, metadata.descriptor.total_work_units);
  write_le(destination, 72uz, metadata.completed_work_units);
  write_le(destination, 80uz, static_cast<std::uint64_t>(payload.size()));
  write_le(destination, 88uz, metadata.descriptor.state_schema);

  const auto payload_destination = destination.subspan(kStateCapsuleHeaderBytes, payload.size());
  if (payload.data() != payload_destination.data())
    std::ranges::copy(payload, payload_destination.begin());
  const std::size_t checksum_offset = kStateCapsuleHeaderBytes + payload.size();
  const ModelDigest checksum = fingerprint_bytes(destination.first(checksum_offset));
  for (std::size_t index{}; index < checksum.size(); ++index)
    destination[checksum_offset + index] = static_cast<std::byte>(checksum[index]);
  return required;
}

std::expected<StateCapsuleView, StateCapsuleError> read_state_capsule(
    std::span<const std::byte> source) noexcept
{
  if (source.size() < kStateCapsuleHeaderBytes + kStateCapsuleChecksumBytes)
    return std::unexpected(
        StateCapsuleError{StateCapsuleErrorCode::kTruncated, "State capsule is truncated"});
  if (!std::ranges::equal(kMagic, source.first(kMagic.size())))
    return std::unexpected(
        StateCapsuleError{StateCapsuleErrorCode::kInvalidHeader, "State capsule magic is invalid"});
  if (read_le<std::uint16_t>(source, 8uz) != kVersion ||
      read_le<std::uint16_t>(source, 10uz) != kStateCapsuleHeaderBytes)
    return std::unexpected(StateCapsuleError{StateCapsuleErrorCode::kUnsupportedVersion,
                                             "State capsule version is unsupported"});
  if (read_le<std::uint32_t>(source, 16uz) != 0u || read_le<std::uint32_t>(source, 20uz) != 0u)
    return std::unexpected(StateCapsuleError{StateCapsuleErrorCode::kInvalidHeader,
                                             "State capsule flags are invalid"});

  const auto payload_u64 = read_le<std::uint64_t>(source, 80uz);
  if (payload_u64 > std::numeric_limits<std::size_t>::max())
    return std::unexpected(StateCapsuleError{StateCapsuleErrorCode::kInvalidMetadata,
                                             "State capsule payload is too large"});
  const auto payload_bytes = static_cast<std::size_t>(payload_u64);
  const std::size_t required = state_capsule_bytes(payload_bytes);
  if (required == 0uz || source.size() != required)
    return std::unexpected(StateCapsuleError{StateCapsuleErrorCode::kTruncated,
                                             "State capsule size is incompatible or truncated"});

  const std::size_t checksum_offset = kStateCapsuleHeaderBytes + payload_bytes;
  const ModelDigest checksum = fingerprint_bytes(source.first(checksum_offset));
  for (std::size_t index{}; index < checksum.size(); ++index) {
    if (source[checksum_offset + index] != static_cast<std::byte>(checksum[index]))
      return std::unexpected(StateCapsuleError{StateCapsuleErrorCode::kChecksumMismatch,
                                               "State capsule checksum mismatch"});
  }

  StateCapsuleMetadata metadata{};
  metadata.descriptor.kind = static_cast<JobKind>(read_le<std::uint16_t>(source, 12uz));
  metadata.state = static_cast<JobState>(read_le<std::uint16_t>(source, 14uz));
  for (std::size_t index{}; index < metadata.descriptor.model_digest.size(); ++index)
    metadata.descriptor.model_digest[index] = std::to_integer<std::uint8_t>(source[24uz + index]);
  metadata.descriptor.session_id = read_le<std::uint64_t>(source, 40uz);
  metadata.descriptor.generation = read_le<std::uint64_t>(source, 48uz);
  metadata.descriptor.deadline_ns = read_le<std::uint64_t>(source, 56uz);
  metadata.descriptor.total_work_units = read_le<std::uint64_t>(source, 64uz);
  metadata.completed_work_units = read_le<std::uint64_t>(source, 72uz);
  metadata.descriptor.state_schema = read_le<std::uint64_t>(source, 88uz);
  if (!valid_progress(metadata))
    return std::unexpected(StateCapsuleError{StateCapsuleErrorCode::kInvalidMetadata,
                                             "State capsule metadata is invalid"});
  return StateCapsuleView{
      .metadata = metadata,
      .payload = source.subspan(kStateCapsuleHeaderBytes, payload_bytes),
  };
}

} // namespace fe::relay
