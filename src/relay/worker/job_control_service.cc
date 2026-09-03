#include "relay/worker/job_control_service.h"

#include <cstddef>
#include <limits>
#include <span>
#include <utility>

namespace fe::relay {
namespace {

[[nodiscard]] bool usable_ring(const SharedMemoryRing& ring, std::size_t slot_bytes) noexcept
{
  const RingConfig config = ring.config();
  return config.capacity >= 2u && config.slot_bytes >= slot_bytes;
}

[[nodiscard]] JobControlCode map_drain_result(WorkerDrainResult result) noexcept
{
  switch (result) {
  case WorkerDrainResult::kStarted:
    return JobControlCode::kSuccess;
  case WorkerDrainResult::kAlreadyDraining:
    return JobControlCode::kAlreadyDraining;
  case WorkerDrainResult::kWouldStrandWork:
    return JobControlCode::kWouldStrandWork;
  case WorkerDrainResult::kInvalidWorker:
    return JobControlCode::kInvalidWorker;
  }
  return JobControlCode::kInvalidRequest;
}

} // namespace

std::expected<JobControlService, std::string> JobControlService::create(
    std::string request_name, std::string response_name, JobWorkerPool& pool,
    const JobTransportStats& transport_stats, std::uint32_t capacity) noexcept
{
  auto requests = SharedMemoryRing::create(std::move(request_name),
                                           RingConfig{.capacity = capacity,
                                                      .slot_bytes = sizeof(JobControlRequest)});
  if (!requests)
    return std::unexpected(requests.error());
  auto responses = SharedMemoryRing::create(std::move(response_name),
                                            RingConfig{.capacity = capacity,
                                                       .slot_bytes = sizeof(JobControlResponse)});
  if (!responses)
    return std::unexpected(responses.error());
  return JobControlService{std::move(*requests), std::move(*responses), pool, transport_stats};
}

std::expected<JobControlService, std::string> JobControlService::connect(
    std::string request_name, std::string response_name, JobWorkerPool& pool,
    const JobTransportStats& transport_stats) noexcept
{
  auto requests = SharedMemoryRing::open(std::move(request_name));
  if (!requests)
    return std::unexpected(requests.error());
  auto responses = SharedMemoryRing::open(std::move(response_name));
  if (!responses)
    return std::unexpected(responses.error());
  if (!usable_ring(*requests, sizeof(JobControlRequest)) ||
      !usable_ring(*responses, sizeof(JobControlResponse)))
    return std::unexpected("Generic job control rings are incompatible with this service build");
  return JobControlService{std::move(*requests), std::move(*responses), pool, transport_stats};
}

void JobControlService::capture_status() noexcept
{
  JobControlStatus& status = response_.metadata.status;
  status = {};
  status.worker_count = static_cast<std::uint32_t>(pool_->worker_count());
  status.accepting_workers = static_cast<std::uint32_t>(pool_->accepting_worker_count());
  status.busy_workers = static_cast<std::uint32_t>(pool_->busy_count());
  status.queued_jobs = static_cast<std::uint32_t>(pool_->queued_count());
  for (std::size_t index{}; index < pool_->worker_count(); ++index) {
    const std::uint32_t bit = std::uint32_t{1u} << index;
    if (pool_->worker_draining(index))
      status.draining_mask |= bit;
    if (pool_->worker_drained(index))
      status.drained_mask |= bit;
    if (pool_->worker_binding(index).bound)
      status.bound_mask |= bit;
  }
  status.worker_failures = pool_->failure_count();
  status.requests_received = transport_stats_->requests_received;
  status.requests_accepted = transport_stats_->requests_accepted;
  status.requests_rejected = transport_stats_->requests_rejected;
  status.results_published = transport_stats_->results_published;
  status.corrupt_inputs = transport_stats_->corrupt_inputs;
  status.publish_blocked = transport_stats_->publish_blocked;
}

void JobControlService::build_response(const JobControlRequest& request) noexcept
{
  response_ = {};
  response_.envelope.struct_size = sizeof(response_);
  response_.envelope.sequence = request.envelope.sequence;
  response_.envelope.session_id = request.envelope.session_id;
  response_.metadata.operation = request.metadata.operation;
  response_.metadata.worker_index = request.metadata.worker_index;

  switch (request.metadata.operation) {
  case JobControlOperation::kStatus:
    break;
  case JobControlOperation::kDrainWorker:
    response_.metadata.code =
        map_drain_result(pool_->request_worker_drain(request.metadata.worker_index));
    break;
  case JobControlOperation::kResumeWorker:
    if (request.metadata.worker_index >= pool_->worker_count())
      response_.metadata.code = JobControlCode::kInvalidWorker;
    else if (!pool_->resume_worker(request.metadata.worker_index))
      response_.metadata.code = JobControlCode::kWorkerNotDrained;
    break;
  case JobControlOperation::kShutdown:
    stop_after_publish_ = true;
    break;
  }
  capture_status();
}

JobControlServiceResult JobControlService::publish_pending() noexcept
{
  switch (responses_.try_push(std::as_bytes(std::span{&response_, 1uz}))) {
  case RingResult::kSuccess:
    response_pending_ = false;
    if (stop_after_publish_)
      stopped_ = true;
    return stopped_ ? JobControlServiceResult::kStopped : JobControlServiceResult::kProgress;
  case RingResult::kFull:
    return JobControlServiceResult::kBackpressured;
  case RingResult::kEmpty:
  case RingResult::kTooLarge:
  case RingResult::kDestinationTooSmall:
  case RingResult::kCorrupt:
    return JobControlServiceResult::kTransportError;
  }
  return JobControlServiceResult::kTransportError;
}

JobControlServiceResult JobControlService::poll() noexcept
{
  if (stopped_)
    return JobControlServiceResult::kStopped;
  if (response_pending_)
    return publish_pending();

  request_ = {};
  std::size_t bytes_read{};
  const RingResult received =
      requests_.try_pop(std::as_writable_bytes(std::span{&request_, 1uz}), bytes_read);
  if (received == RingResult::kEmpty)
    return JobControlServiceResult::kIdle;
  if (received == RingResult::kCorrupt)
    return JobControlServiceResult::kCorruptInput;
  if (received != RingResult::kSuccess)
    return JobControlServiceResult::kTransportError;
  if (bytes_read != sizeof(request_) || validate(request_) != ProtocolResult::kSuccess)
    return JobControlServiceResult::kCorruptInput;

  build_response(request_);
  response_pending_ = true;
  return publish_pending();
}

std::string_view to_string(JobControlServiceResult result) noexcept
{
  switch (result) {
  case JobControlServiceResult::kIdle:
    return "idle";
  case JobControlServiceResult::kProgress:
    return "progress";
  case JobControlServiceResult::kBackpressured:
    return "backpressured";
  case JobControlServiceResult::kStopped:
    return "stopped";
  case JobControlServiceResult::kCorruptInput:
    return "corrupt_input";
  case JobControlServiceResult::kTransportError:
    return "transport_error";
  }
  return "unknown";
}

} // namespace fe::relay
