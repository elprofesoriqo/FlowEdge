#pragma once

#include "relay/jobs/job_types.h"
#include "relay/protocol/messages.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <span>
#include <string_view>
#include <type_traits>

#ifndef FLOWEDGE_RELAY_MAX_JOB_PAYLOAD_BYTES
#define FLOWEDGE_RELAY_MAX_JOB_PAYLOAD_BYTES 65536
#endif

namespace fe::relay {

inline constexpr std::size_t kMaxJobPayloadBytes = FLOWEDGE_RELAY_MAX_JOB_PAYLOAD_BYTES;
static_assert(kMaxJobPayloadBytes > 0uz);
static_assert(kMaxJobPayloadBytes <= std::numeric_limits<std::uint32_t>::max());

enum class JobResultCode : std::uint32_t
{
  kProgress = 0u,
  kComplete = 1u,
  kCancelled = 2u,
  kFailed = 3u,
  kRejectedStale = 4u,
  kRejectedDeadline = 5u,
  kRejectedCapacity = 6u,
  kAdapterNotFound = 7u,
};

struct JobRequestMetadata
{
  std::uint32_t struct_size{};
  std::uint32_t reserved{};
  JobDescriptor descriptor{};
  std::uint64_t timestamp_ns{};
  std::uint64_t payload_bytes{};
};

struct JobResultMetadata
{
  std::uint32_t struct_size{};
  std::uint32_t reserved{};
  JobDescriptor descriptor{};
  JobProgress progress{};
  std::uint64_t timestamp_ns{};
  std::uint64_t payload_bytes{};
};

struct JobRequestMessage
{
  MessageEnvelope envelope{.kind = MessageKind::kJobRequest};
  JobRequestMetadata metadata{};
  std::array<std::byte, kMaxJobPayloadBytes> payload{};

  [[nodiscard]] std::span<std::byte> payload_values() noexcept
  {
    return {payload.data(), static_cast<std::size_t>(std::min<std::uint64_t>(metadata.payload_bytes,
                                                                             kMaxJobPayloadBytes))};
  }
  [[nodiscard]] std::span<const std::byte> payload_values() const noexcept
  {
    return {payload.data(), static_cast<std::size_t>(std::min<std::uint64_t>(metadata.payload_bytes,
                                                                             kMaxJobPayloadBytes))};
  }
};

struct JobResultMessage
{
  MessageEnvelope envelope{.kind = MessageKind::kJobResult};
  JobResultMetadata metadata{};
  std::array<std::byte, kMaxJobPayloadBytes> payload{};

  [[nodiscard]] std::span<std::byte> payload_values() noexcept
  {
    return {payload.data(), static_cast<std::size_t>(std::min<std::uint64_t>(metadata.payload_bytes,
                                                                             kMaxJobPayloadBytes))};
  }
  [[nodiscard]] std::span<const std::byte> payload_values() const noexcept
  {
    return {payload.data(), static_cast<std::size_t>(std::min<std::uint64_t>(metadata.payload_bytes,
                                                                             kMaxJobPayloadBytes))};
  }
};

static_assert(sizeof(JobRequestMessage) <= std::numeric_limits<std::uint32_t>::max());
static_assert(sizeof(JobResultMessage) <= std::numeric_limits<std::uint32_t>::max());

[[nodiscard]] constexpr std::size_t wire_size(const JobRequestMessage& message) noexcept
{
  return offsetof(JobRequestMessage, payload) +
         static_cast<std::size_t>(
             std::min<std::uint64_t>(message.metadata.payload_bytes, kMaxJobPayloadBytes));
}

[[nodiscard]] constexpr std::size_t wire_size(const JobResultMessage& message) noexcept
{
  return offsetof(JobResultMessage, payload) +
         static_cast<std::size_t>(
             std::min<std::uint64_t>(message.metadata.payload_bytes, kMaxJobPayloadBytes));
}

[[nodiscard]] inline std::span<const std::byte> wire_bytes(
    const JobRequestMessage& message) noexcept
{
  return std::as_bytes(std::span{&message, 1uz}).first(wire_size(message));
}

[[nodiscard]] inline std::span<const std::byte> wire_bytes(const JobResultMessage& message) noexcept
{
  return std::as_bytes(std::span{&message, 1uz}).first(wire_size(message));
}

[[nodiscard]] constexpr bool valid_job_result_code(JobResultCode code) noexcept
{
  return code >= JobResultCode::kProgress && code <= JobResultCode::kAdapterNotFound;
}

[[nodiscard]] constexpr bool compatible(JobResultCode code, JobState state) noexcept
{
  if (code == JobResultCode::kProgress)
    return state == JobState::kReady || state == JobState::kRunning;
  if (code == JobResultCode::kComplete)
    return state == JobState::kComplete;
  if (code == JobResultCode::kCancelled)
    return state == JobState::kCancelled;
  return state == JobState::kFailed;
}

[[nodiscard]] constexpr JobResultCode result_code(JobState state) noexcept
{
  if (state == JobState::kComplete)
    return JobResultCode::kComplete;
  if (state == JobState::kCancelled)
    return JobResultCode::kCancelled;
  if (state == JobState::kFailed)
    return JobResultCode::kFailed;
  return JobResultCode::kProgress;
}

[[nodiscard]] constexpr JobResultCode job_result_code(const JobResultMessage& message) noexcept
{
  return static_cast<JobResultCode>(message.envelope.flags);
}

[[nodiscard]] constexpr std::string_view to_string(JobResultCode code) noexcept
{
  switch (code) {
  case JobResultCode::kProgress:
    return "progress";
  case JobResultCode::kComplete:
    return "complete";
  case JobResultCode::kCancelled:
    return "cancelled";
  case JobResultCode::kFailed:
    return "failed";
  case JobResultCode::kRejectedStale:
    return "rejected_stale";
  case JobResultCode::kRejectedDeadline:
    return "rejected_deadline";
  case JobResultCode::kRejectedCapacity:
    return "rejected_capacity";
  case JobResultCode::kAdapterNotFound:
    return "adapter_not_found";
  }
  return "unknown";
}

[[nodiscard]] inline ProtocolResult validate(const JobRequestMessage& message) noexcept
{
  if (message.envelope.magic != kMessageMagic || message.envelope.version != kMessageVersion ||
      message.envelope.kind != MessageKind::kJobRequest || message.envelope.flags != 0u ||
      message.envelope.struct_size != wire_size(message))
    return ProtocolResult::kInvalidEnvelope;
  if (message.metadata.payload_bytes > kMaxJobPayloadBytes)
    return ProtocolResult::kDimensionExceeded;
  if (message.metadata.struct_size != sizeof(JobRequestMetadata) ||
      message.metadata.reserved != 0u || !valid_job_descriptor(message.metadata.descriptor) ||
      message.envelope.session_id != message.metadata.descriptor.session_id ||
      (message.metadata.descriptor.deadline_ns != 0u &&
       message.metadata.descriptor.deadline_ns < message.metadata.timestamp_ns))
    return ProtocolResult::kInvalidMetadata;
  return ProtocolResult::kSuccess;
}

[[nodiscard]] inline ProtocolResult validate(const JobResultMessage& message) noexcept
{
  const JobResultCode code = job_result_code(message);
  if (message.envelope.magic != kMessageMagic || message.envelope.version != kMessageVersion ||
      message.envelope.kind != MessageKind::kJobResult || !valid_job_result_code(code) ||
      message.envelope.struct_size != wire_size(message))
    return ProtocolResult::kInvalidEnvelope;
  if (message.metadata.payload_bytes > kMaxJobPayloadBytes)
    return ProtocolResult::kDimensionExceeded;
  if (message.metadata.struct_size != sizeof(JobResultMetadata) ||
      message.metadata.reserved != 0u || !valid_job_descriptor(message.metadata.descriptor) ||
      message.envelope.session_id != message.metadata.descriptor.session_id ||
      !valid_job_progress(message.metadata.descriptor, message.metadata.progress) ||
      !compatible(code, message.metadata.progress.state))
    return ProtocolResult::kInvalidMetadata;
  return ProtocolResult::kSuccess;
}

[[nodiscard]] inline ProtocolResult make_job_request(JobRequestMessage& destination,
                                                     std::uint64_t sequence,
                                                     JobDescriptor descriptor,
                                                     std::uint64_t timestamp_ns,
                                                     std::span<const std::byte> payload) noexcept
{
  if (!valid_job_descriptor(descriptor) ||
      (descriptor.deadline_ns != 0u && descriptor.deadline_ns < timestamp_ns))
    return ProtocolResult::kInvalidMetadata;
  if (payload.size() > kMaxJobPayloadBytes)
    return ProtocolResult::kDimensionExceeded;

  destination.envelope = {};
  destination.metadata = {};
  destination.envelope.kind = MessageKind::kJobRequest;
  destination.envelope.sequence = sequence;
  destination.envelope.session_id = descriptor.session_id;
  destination.metadata.struct_size = sizeof(JobRequestMetadata);
  destination.metadata.descriptor = descriptor;
  destination.metadata.timestamp_ns = timestamp_ns;
  destination.metadata.payload_bytes = payload.size();
  destination.envelope.struct_size = static_cast<std::uint32_t>(wire_size(destination));
  std::ranges::copy(payload, destination.payload.begin());
  return ProtocolResult::kSuccess;
}

[[nodiscard]] inline ProtocolResult make_job_result(
    JobResultMessage& destination, std::uint64_t sequence, JobDescriptor descriptor,
    JobProgress progress, std::uint64_t timestamp_ns, JobResultCode code,
    std::span<const std::byte> payload = {}) noexcept
{
  if (!valid_job_descriptor(descriptor) || !valid_job_progress(descriptor, progress) ||
      !valid_job_result_code(code) || !compatible(code, progress.state))
    return ProtocolResult::kInvalidMetadata;
  if (payload.size() > kMaxJobPayloadBytes)
    return ProtocolResult::kDimensionExceeded;

  destination.envelope = {};
  destination.metadata = {};
  destination.envelope.kind = MessageKind::kJobResult;
  destination.envelope.flags = static_cast<std::uint32_t>(code);
  destination.envelope.sequence = sequence;
  destination.envelope.session_id = descriptor.session_id;
  destination.metadata.struct_size = sizeof(JobResultMetadata);
  destination.metadata.descriptor = descriptor;
  destination.metadata.progress = progress;
  destination.metadata.timestamp_ns = timestamp_ns;
  destination.metadata.payload_bytes = payload.size();
  destination.envelope.struct_size = static_cast<std::uint32_t>(wire_size(destination));
  std::ranges::copy(payload, destination.payload.begin());
  return ProtocolResult::kSuccess;
}

static_assert(std::is_trivially_copyable_v<JobRequestMetadata>);
static_assert(std::is_trivially_copyable_v<JobResultMetadata>);
static_assert(std::is_trivially_copyable_v<JobRequestMessage>);
static_assert(std::is_trivially_copyable_v<JobResultMessage>);

} // namespace fe::relay
