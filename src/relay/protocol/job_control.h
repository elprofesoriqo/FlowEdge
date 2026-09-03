#pragma once

#include "relay/protocol/messages.h"

#include <cstdint>
#include <limits>
#include <string_view>
#include <type_traits>

namespace fe::relay {

inline constexpr std::uint32_t kJobControlProtocolVersion = 2u;
inline constexpr std::uint32_t kAllJobWorkers = std::numeric_limits<std::uint32_t>::max();
inline constexpr std::uint32_t kMaxJobControlWorkers = 8u;

enum class JobControlOperation : std::uint32_t
{
  kStatus = 1u,
  kDrainWorker = 2u,
  kResumeWorker = 3u,
  kShutdown = 4u,
  kRecoverWorker = 5u,
};

enum class JobControlCode : std::uint32_t
{
  kSuccess = 0u,
  kInvalidRequest = 1u,
  kInvalidWorker = 2u,
  kAlreadyDraining = 3u,
  kWouldStrandWork = 4u,
  kWorkerNotDrained = 5u,
  kWorkerQuarantined = 6u,
  kWorkerNotQuarantined = 7u,
};

struct JobControlRequestMetadata
{
  std::uint32_t struct_size{sizeof(JobControlRequestMetadata)};
  std::uint32_t protocol_version{kJobControlProtocolVersion};
  JobControlOperation operation{JobControlOperation::kStatus};
  std::uint32_t worker_index{kAllJobWorkers};
  std::uint64_t reserved{};
};

struct JobControlStatus
{
  std::uint32_t worker_count{};
  std::uint32_t accepting_workers{};
  std::uint32_t busy_workers{};
  std::uint32_t queued_jobs{};
  std::uint32_t draining_mask{};
  std::uint32_t drained_mask{};
  std::uint32_t bound_mask{};
  std::uint32_t quarantined_mask{};
  std::uint64_t worker_failures{};
  std::uint64_t worker_quarantines{};
  std::uint64_t qos_rejections{};
  std::uint64_t requests_received{};
  std::uint64_t requests_accepted{};
  std::uint64_t requests_rejected{};
  std::uint64_t results_published{};
  std::uint64_t corrupt_inputs{};
  std::uint64_t publish_blocked{};
};

struct JobControlResponseMetadata
{
  std::uint32_t struct_size{sizeof(JobControlResponseMetadata)};
  std::uint32_t protocol_version{kJobControlProtocolVersion};
  JobControlOperation operation{JobControlOperation::kStatus};
  JobControlCode code{JobControlCode::kSuccess};
  std::uint32_t worker_index{kAllJobWorkers};
  std::uint32_t reserved{};
  JobControlStatus status{};
};

struct JobControlRequest
{
  MessageEnvelope envelope{.kind = MessageKind::kJobControlRequest};
  JobControlRequestMetadata metadata{};
};

struct JobControlResponse
{
  MessageEnvelope envelope{.kind = MessageKind::kJobControlResponse};
  JobControlResponseMetadata metadata{};
};

[[nodiscard]] constexpr bool valid_job_control_operation(JobControlOperation operation) noexcept
{
  return operation >= JobControlOperation::kStatus &&
         operation <= JobControlOperation::kRecoverWorker;
}

[[nodiscard]] constexpr bool valid_job_control_code(JobControlCode code) noexcept
{
  return code >= JobControlCode::kSuccess && code <= JobControlCode::kWorkerNotQuarantined;
}

[[nodiscard]] constexpr bool requires_worker(JobControlOperation operation) noexcept
{
  return operation == JobControlOperation::kDrainWorker ||
         operation == JobControlOperation::kResumeWorker ||
         operation == JobControlOperation::kRecoverWorker;
}

[[nodiscard]] constexpr bool valid_worker_target(JobControlOperation operation,
                                                 std::uint32_t worker_index) noexcept
{
  const bool has_worker = worker_index != kAllJobWorkers;
  return requires_worker(operation) == has_worker;
}

[[nodiscard]] constexpr ProtocolResult validate(const JobControlRequest& request) noexcept
{
  if (!valid_envelope(request.envelope, MessageKind::kJobControlRequest, sizeof(request)))
    return ProtocolResult::kInvalidEnvelope;
  if (request.metadata.struct_size != sizeof(JobControlRequestMetadata) ||
      request.metadata.protocol_version != kJobControlProtocolVersion ||
      !valid_job_control_operation(request.metadata.operation) || request.metadata.reserved != 0u ||
      !valid_worker_target(request.metadata.operation, request.metadata.worker_index))
    return ProtocolResult::kInvalidMetadata;
  return ProtocolResult::kSuccess;
}

[[nodiscard]] constexpr ProtocolResult validate(const JobControlResponse& response) noexcept
{
  if (!valid_envelope(response.envelope, MessageKind::kJobControlResponse, sizeof(response)))
    return ProtocolResult::kInvalidEnvelope;
  if (response.metadata.struct_size != sizeof(JobControlResponseMetadata) ||
      response.metadata.protocol_version != kJobControlProtocolVersion ||
      !valid_job_control_operation(response.metadata.operation) ||
      !valid_job_control_code(response.metadata.code) || response.metadata.reserved != 0u ||
      !valid_worker_target(response.metadata.operation, response.metadata.worker_index))
    return ProtocolResult::kInvalidMetadata;
  const JobControlStatus& status = response.metadata.status;
  if (status.worker_count > kMaxJobControlWorkers ||
      status.accepting_workers > status.worker_count || status.busy_workers > status.worker_count)
    return ProtocolResult::kInvalidMetadata;
  const std::uint32_t valid_mask =
      status.worker_count == 0u ? 0u : (std::uint32_t{1u} << status.worker_count) - 1u;
  if (((status.draining_mask | status.drained_mask | status.bound_mask | status.quarantined_mask) &
       ~valid_mask) != 0u ||
      (status.drained_mask & ~status.draining_mask) != 0u ||
      (status.quarantined_mask & ~status.draining_mask) != 0u)
    return ProtocolResult::kInvalidMetadata;
  return ProtocolResult::kSuccess;
}

[[nodiscard]] constexpr JobControlRequest make_job_control_request(
    std::uint64_t sequence, std::uint64_t session_id, JobControlOperation operation,
    std::uint32_t worker_index = kAllJobWorkers) noexcept
{
  JobControlRequest request{};
  request.envelope.struct_size = sizeof(request);
  request.envelope.sequence = sequence;
  request.envelope.session_id = session_id;
  request.metadata.operation = operation;
  request.metadata.worker_index = worker_index;
  return request;
}

[[nodiscard]] constexpr std::string_view to_string(JobControlOperation operation) noexcept
{
  switch (operation) {
  case JobControlOperation::kStatus:
    return "status";
  case JobControlOperation::kDrainWorker:
    return "drain";
  case JobControlOperation::kResumeWorker:
    return "resume";
  case JobControlOperation::kShutdown:
    return "shutdown";
  case JobControlOperation::kRecoverWorker:
    return "recover";
  }
  return "unknown";
}

[[nodiscard]] constexpr std::string_view to_string(JobControlCode code) noexcept
{
  switch (code) {
  case JobControlCode::kSuccess:
    return "success";
  case JobControlCode::kInvalidRequest:
    return "invalid_request";
  case JobControlCode::kInvalidWorker:
    return "invalid_worker";
  case JobControlCode::kAlreadyDraining:
    return "already_draining";
  case JobControlCode::kWouldStrandWork:
    return "would_strand_work";
  case JobControlCode::kWorkerNotDrained:
    return "worker_not_drained";
  case JobControlCode::kWorkerQuarantined:
    return "worker_quarantined";
  case JobControlCode::kWorkerNotQuarantined:
    return "worker_not_quarantined";
  }
  return "unknown";
}

static_assert(std::is_trivially_copyable_v<JobControlRequest>);
static_assert(std::is_trivially_copyable_v<JobControlResponse>);

} // namespace fe::relay
