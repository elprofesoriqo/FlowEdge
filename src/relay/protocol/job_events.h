#pragma once

#include "relay/protocol/job_messages.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <type_traits>

namespace fe::relay {

inline constexpr std::uint16_t kJobEventVersion = 1u;
inline constexpr std::uint32_t kNoWorker = std::numeric_limits<std::uint32_t>::max();

enum class JobEventKind : std::uint16_t
{
  kUnknown = 0u,
  kAdmitted = 1u,
  kDispatched = 2u,
  kStarted = 3u,
  kProgress = 4u,
  kPreempted = 5u,
  kMigrationStarted = 6u,
  kMigrationCompleted = 7u,
  kCompleted = 8u,
  kCancelled = 9u,
  kFailed = 10u,
  kRejected = 11u,
};

struct JobEventTiming
{
  std::uint64_t queue_ns{};
  std::uint64_t execution_ns{};
  std::uint64_t end_to_end_ns{};
  std::uint64_t cancellation_ns{};
};

struct JobEventMetadata
{
  std::uint32_t struct_size{};
  std::uint16_t protocol_version{kJobEventVersion};
  JobEventKind event{};
  JobDescriptor descriptor{};
  JobProgress progress{};
  std::uint64_t timestamp_ns{};
  JobEventTiming timing{};
  std::uint64_t related_generation{};
  std::uint32_t worker_index{kNoWorker};
  std::uint32_t peer_worker_index{kNoWorker};
  std::uint32_t queue_depth{};
  JobResultCode result_code{JobResultCode::kProgress};
  JobServiceClass service_class{JobServiceClass::kBestEffort};
  std::uint32_t reserved2{};
};

struct JobEventMessage
{
  MessageEnvelope envelope{.kind = MessageKind::kJobEvent};
  JobEventMetadata metadata{};
};

struct JobEventDetails
{
  JobEventKind event{};
  JobDescriptor descriptor{};
  JobProgress progress{};
  std::uint64_t timestamp_ns{};
  JobEventTiming timing{};
  std::uint64_t related_generation{};
  std::uint32_t worker_index{kNoWorker};
  std::uint32_t peer_worker_index{kNoWorker};
  std::uint32_t queue_depth{};
  JobResultCode result_code{JobResultCode::kProgress};
  JobServiceClass service_class{JobServiceClass::kBestEffort};
};

[[nodiscard]] constexpr bool valid_job_event_kind(JobEventKind event) noexcept
{
  return event >= JobEventKind::kAdmitted && event <= JobEventKind::kRejected;
}

[[nodiscard]] constexpr bool rejected(JobResultCode code) noexcept
{
  return code == JobResultCode::kRejectedStale || code == JobResultCode::kRejectedDeadline ||
         code == JobResultCode::kRejectedCapacity || code == JobResultCode::kAdapterNotFound ||
         code == JobResultCode::kInvalidRequest || code == JobResultCode::kRejectedQos;
}

[[nodiscard]] constexpr bool compatible(JobEventKind event, JobProgress progress,
                                        JobResultCode code) noexcept
{
  switch (event) {
  case JobEventKind::kUnknown:
    return false;
  case JobEventKind::kAdmitted:
  case JobEventKind::kDispatched:
    return progress.state == JobState::kReady && code == JobResultCode::kProgress;
  case JobEventKind::kStarted:
  case JobEventKind::kProgress:
  case JobEventKind::kPreempted:
  case JobEventKind::kMigrationStarted:
  case JobEventKind::kMigrationCompleted:
    return (progress.state == JobState::kReady || progress.state == JobState::kRunning) &&
           code == JobResultCode::kProgress;
  case JobEventKind::kCompleted:
    return progress.state == JobState::kComplete && code == JobResultCode::kComplete;
  case JobEventKind::kCancelled:
    return progress.state == JobState::kCancelled && code == JobResultCode::kCancelled;
  case JobEventKind::kFailed:
    return progress.state == JobState::kFailed && code == JobResultCode::kFailed;
  case JobEventKind::kRejected:
    return progress.state == JobState::kFailed && rejected(code);
  }
  return false;
}

[[nodiscard]] constexpr std::string_view to_string(JobEventKind event) noexcept
{
  switch (event) {
  case JobEventKind::kUnknown:
    break;
  case JobEventKind::kAdmitted:
    return "admitted";
  case JobEventKind::kDispatched:
    return "dispatched";
  case JobEventKind::kStarted:
    return "started";
  case JobEventKind::kProgress:
    return "progress";
  case JobEventKind::kPreempted:
    return "preempted";
  case JobEventKind::kMigrationStarted:
    return "migration_started";
  case JobEventKind::kMigrationCompleted:
    return "migration_completed";
  case JobEventKind::kCompleted:
    return "completed";
  case JobEventKind::kCancelled:
    return "cancelled";
  case JobEventKind::kFailed:
    return "failed";
  case JobEventKind::kRejected:
    return "rejected";
  }
  return "unknown";
}

[[nodiscard]] constexpr std::size_t wire_size(const JobEventMessage&) noexcept
{
  return sizeof(JobEventMessage);
}

[[nodiscard]] inline std::span<const std::byte> wire_bytes(const JobEventMessage& message) noexcept
{
  return std::as_bytes(std::span{&message, 1uz});
}

[[nodiscard]] inline ProtocolResult validate(const JobEventMessage& message) noexcept
{
  if (!valid_envelope(message.envelope, MessageKind::kJobEvent, sizeof(message)))
    return ProtocolResult::kInvalidEnvelope;
  const JobEventMetadata& metadata = message.metadata;
  if (metadata.struct_size != sizeof(JobEventMetadata) ||
      metadata.protocol_version != kJobEventVersion || metadata.reserved2 != 0u ||
      !valid_job_event_kind(metadata.event) || !valid_job_service_class(metadata.service_class) ||
      !valid_job_descriptor(metadata.descriptor) ||
      !valid_job_progress(metadata.descriptor, metadata.progress) ||
      message.envelope.session_id != metadata.descriptor.session_id ||
      !compatible(metadata.event, metadata.progress, metadata.result_code))
    return ProtocolResult::kInvalidMetadata;
  return ProtocolResult::kSuccess;
}

[[nodiscard]] inline ProtocolResult make_job_event(JobEventMessage& destination,
                                                   std::uint64_t sequence,
                                                   const JobEventDetails& details) noexcept
{
  destination = {};
  destination.envelope.struct_size = sizeof(destination);
  destination.envelope.sequence = sequence;
  destination.envelope.session_id = details.descriptor.session_id;
  destination.metadata.struct_size = sizeof(JobEventMetadata);
  destination.metadata.protocol_version = kJobEventVersion;
  destination.metadata.event = details.event;
  destination.metadata.descriptor = details.descriptor;
  destination.metadata.progress = details.progress;
  destination.metadata.timestamp_ns = details.timestamp_ns;
  destination.metadata.timing = details.timing;
  destination.metadata.related_generation = details.related_generation;
  destination.metadata.worker_index = details.worker_index;
  destination.metadata.peer_worker_index = details.peer_worker_index;
  destination.metadata.queue_depth = details.queue_depth;
  destination.metadata.result_code = details.result_code;
  destination.metadata.service_class = details.service_class;
  return validate(destination);
}

static_assert(sizeof(JobEventMetadata) == 168uz);
static_assert(sizeof(JobEventMessage) == 200uz);
static_assert(std::is_trivially_copyable_v<JobEventMetadata>);
static_assert(std::is_trivially_copyable_v<JobEventMessage>);

} // namespace fe::relay
