#include "relay/worker/job_service.h"

#include <chrono>
#include <cstddef>
#include <cstring>
#include <exception>
#include <memory>
#include <span>
#include <utility>

namespace fe::relay {
namespace {

[[nodiscard]] std::uint64_t monotonic_ns() noexcept
{
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now().time_since_epoch())
                                        .count());
}

[[nodiscard]] bool usable_ring(const SharedMemoryRing& ring, std::size_t slot_bytes) noexcept
{
  const RingConfig config = ring.config();
  return config.capacity >= 2u && config.slot_bytes >= slot_bytes;
}

[[nodiscard]] bool accepted(JobSubmitResult result) noexcept
{
  return result == JobSubmitResult::kAccepted;
}

} // namespace

std::expected<JobService, std::string> JobService::create(std::string request_name,
                                                          std::string result_name,
                                                          JobWorkerPool& pool,
                                                          std::uint32_t capacity) noexcept
{
  auto requests = SharedMemoryRing::create(std::move(request_name),
                                           RingConfig{.capacity = capacity,
                                                      .slot_bytes = sizeof(JobRequestMessage)});
  if (!requests)
    return std::unexpected(requests.error());
  auto results = SharedMemoryRing::create(std::move(result_name),
                                          RingConfig{.capacity = capacity,
                                                     .slot_bytes = sizeof(JobResultMessage)});
  if (!results)
    return std::unexpected(results.error());
  try {
    return JobService{std::move(*requests), std::move(*results), pool,
                      std::make_unique<JobRequestMessage>(), std::make_unique<JobResultMessage>()};
  } catch (const std::exception& error) {
    return std::unexpected("Failed to allocate generic job service buffers: " +
                           std::string{error.what()});
  } catch (...) {
    return std::unexpected("Failed to allocate generic job service buffers");
  }
}

std::expected<JobService, std::string> JobService::connect(std::string request_name,
                                                           std::string result_name,
                                                           JobWorkerPool& pool) noexcept
{
  auto requests = SharedMemoryRing::open(std::move(request_name));
  if (!requests)
    return std::unexpected(requests.error());
  auto results = SharedMemoryRing::open(std::move(result_name));
  if (!results)
    return std::unexpected(results.error());
  if (!usable_ring(*requests, sizeof(JobRequestMessage)) ||
      !usable_ring(*results, sizeof(JobResultMessage)))
    return std::unexpected(
        "Generic job ring slot capacity is incompatible with this service build");
  try {
    return JobService{std::move(*requests), std::move(*results), pool,
                      std::make_unique<JobRequestMessage>(), std::make_unique<JobResultMessage>()};
  } catch (const std::exception& error) {
    return std::unexpected("Failed to allocate generic job service buffers: " +
                           std::string{error.what()});
  } catch (...) {
    return std::unexpected("Failed to allocate generic job service buffers");
  }
}

JobServiceResult JobService::publish(const JobResultMessage& result) noexcept
{
  switch (results_.try_push(wire_bytes(result))) {
  case RingResult::kSuccess:
    ++stats_.results_published;
    return JobServiceResult::kProgress;
  case RingResult::kFull:
    ++stats_.publish_blocked;
    return JobServiceResult::kBackpressured;
  case RingResult::kEmpty:
  case RingResult::kTooLarge:
  case RingResult::kDestinationTooSmall:
  case RingResult::kCorrupt:
    return JobServiceResult::kTransportError;
  }
  return JobServiceResult::kTransportError;
}

JobServiceResult JobService::poll(std::uint64_t now_ns) noexcept
{
  if (stopped_)
    return JobServiceResult::kStopped;

  bool made_progress{false};
  if (rejection_pending_) {
    const JobServiceResult published = publish(*rejection_);
    if (published != JobServiceResult::kProgress)
      return published;
    rejection_pending_ = false;
    made_progress = true;
  }

  if (const JobResultMessage* result = pool_->ready_result()) {
    const JobServiceResult published = publish(*result);
    if (published != JobServiceResult::kProgress)
      return published;
    pool_->release_ready_result();
    made_progress = true;
  }

  request_->envelope = {};
  request_->metadata = {};
  std::size_t bytes_read{};
  const RingResult received =
      requests_.try_pop(std::as_writable_bytes(std::span{request_.get(), 1uz}), bytes_read);
  if (received == RingResult::kEmpty)
    return made_progress ? JobServiceResult::kProgress : JobServiceResult::kIdle;
  if (received == RingResult::kCorrupt) {
    ++stats_.corrupt_inputs;
    return JobServiceResult::kCorruptInput;
  }
  if (received != RingResult::kSuccess)
    return JobServiceResult::kTransportError;
  if (bytes_read < sizeof(MessageEnvelope)) {
    ++stats_.corrupt_inputs;
    return JobServiceResult::kCorruptInput;
  }

  if (request_->envelope.kind == MessageKind::kShutdown) {
    ControlMessage control{};
    if (bytes_read != sizeof(control)) {
      ++stats_.corrupt_inputs;
      return JobServiceResult::kCorruptInput;
    }
    std::memcpy(&control, request_.get(), sizeof(control));
    if (!valid_envelope(control.envelope, MessageKind::kShutdown, sizeof(control))) {
      ++stats_.corrupt_inputs;
      return JobServiceResult::kCorruptInput;
    }
    stopped_ = true;
    return JobServiceResult::kStopped;
  }

  if (request_->envelope.kind != MessageKind::kJobRequest || bytes_read != wire_size(*request_) ||
      validate(*request_) != ProtocolResult::kSuccess) {
    ++stats_.corrupt_inputs;
    return JobServiceResult::kCorruptInput;
  }

  ++stats_.requests_received;
  const JobSubmitResult submitted =
      pool_->submit(*request_, now_ns == 0u ? monotonic_ns() : now_ns, rejection_.get());
  if (accepted(submitted)) {
    ++stats_.requests_accepted;
    return JobServiceResult::kProgress;
  }

  ++stats_.requests_rejected;
  if (validate(*rejection_) != ProtocolResult::kSuccess)
    return JobServiceResult::kTransportError;
  rejection_pending_ = true;
  const JobServiceResult published = publish(*rejection_);
  if (published == JobServiceResult::kProgress)
    rejection_pending_ = false;
  return published;
}

std::string_view to_string(JobServiceResult result) noexcept
{
  switch (result) {
  case JobServiceResult::kIdle:
    return "idle";
  case JobServiceResult::kProgress:
    return "progress";
  case JobServiceResult::kBackpressured:
    return "backpressured";
  case JobServiceResult::kStopped:
    return "stopped";
  case JobServiceResult::kCorruptInput:
    return "corrupt input";
  case JobServiceResult::kTransportError:
    return "transport error";
  }
  return "unknown";
}

} // namespace fe::relay
