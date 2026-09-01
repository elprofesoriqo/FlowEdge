#pragma once

#include "protocol/model_identity.h"

#include <cstdint>
#include <string_view>

namespace fe::relay {

inline constexpr std::uint32_t kCooperativeJobVersion = 1u;

enum class JobKind : std::uint16_t
{
  kUnknown = 0u,
  kIterative = 1u,
  kStreaming = 2u,
  kSpeculative = 3u,
};

enum class JobState : std::uint16_t
{
  kReady = 0u,
  kRunning = 1u,
  kComplete = 2u,
  kCancelled = 3u,
  kFailed = 4u,
};

struct JobDescriptor
{
  std::uint32_t protocol_version{kCooperativeJobVersion};
  JobKind kind{JobKind::kUnknown};
  ModelDigest model_digest{};
  std::uint64_t state_schema{};
  std::uint64_t session_id{};
  std::uint64_t generation{};
  std::uint64_t deadline_ns{};
  std::uint64_t total_work_units{};
};

struct JobProgress
{
  JobState state{JobState::kReady};
  std::uint64_t completed_work_units{};
  std::uint64_t remaining_work_units{};
};

enum class JobErrorCode : std::uint8_t
{
  kInvalidDescriptor,
  kInvalidState,
  kInvalidBudget,
  kCallbackFailed,
  kCallbackContract,
  kCapsuleInvalid,
  kCapsuleIncompatible,
  kBufferTooSmall,
};

struct JobError
{
  JobErrorCode code{};
  std::string_view message{};
};

[[nodiscard]] constexpr bool valid_job_kind(JobKind kind) noexcept
{
  return kind == JobKind::kIterative || kind == JobKind::kStreaming ||
         kind == JobKind::kSpeculative;
}

[[nodiscard]] constexpr bool valid_job_state(JobState state) noexcept
{
  return state >= JobState::kReady && state <= JobState::kFailed;
}

[[nodiscard]] constexpr std::string_view to_string(JobKind kind) noexcept
{
  switch (kind) {
  case JobKind::kIterative:
    return "iterative";
  case JobKind::kStreaming:
    return "streaming";
  case JobKind::kSpeculative:
    return "speculative";
  case JobKind::kUnknown:
    break;
  }
  return "unknown";
}

[[nodiscard]] constexpr std::string_view to_string(JobState state) noexcept
{
  switch (state) {
  case JobState::kReady:
    return "ready";
  case JobState::kRunning:
    return "running";
  case JobState::kComplete:
    return "complete";
  case JobState::kCancelled:
    return "cancelled";
  case JobState::kFailed:
    return "failed";
  }
  return "unknown";
}

[[nodiscard]] constexpr bool terminal(JobState state) noexcept
{
  return state == JobState::kComplete || state == JobState::kCancelled ||
         state == JobState::kFailed;
}

[[nodiscard]] constexpr bool valid_job_progress(const JobDescriptor& descriptor,
                                                const JobProgress& progress) noexcept
{
  if (!valid_job_state(progress.state) ||
      progress.completed_work_units > descriptor.total_work_units ||
      progress.remaining_work_units != descriptor.total_work_units - progress.completed_work_units)
    return false;
  if (progress.state == JobState::kReady)
    return progress.completed_work_units == 0u;
  if (progress.state == JobState::kRunning)
    return progress.completed_work_units < descriptor.total_work_units;
  if (progress.state == JobState::kComplete)
    return progress.completed_work_units == descriptor.total_work_units;
  return true;
}

[[nodiscard]] constexpr bool valid_job_descriptor(const JobDescriptor& descriptor) noexcept
{
  bool has_digest{false};
  for (const std::uint8_t byte : descriptor.model_digest)
    has_digest = has_digest || byte != 0u;
  return descriptor.protocol_version == kCooperativeJobVersion && valid_job_kind(descriptor.kind) &&
         descriptor.state_schema != 0u && descriptor.total_work_units != 0u && has_digest;
}

} // namespace fe::relay
