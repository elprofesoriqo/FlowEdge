#include "relay/jobs/cooperative_job.h"

#include "relay/jobs/state_capsule.h"

#include <algorithm>
#include <limits>

namespace fe::relay {
namespace {

[[nodiscard]] bool same_descriptor(const JobDescriptor& left, const JobDescriptor& right) noexcept
{
  return left.protocol_version == right.protocol_version && left.kind == right.kind &&
         left.model_digest == right.model_digest && left.state_schema == right.state_schema &&
         left.session_id == right.session_id && left.generation == right.generation &&
         left.deadline_ns == right.deadline_ns && left.total_work_units == right.total_work_units;
}

} // namespace

std::expected<CooperativeJob, JobError> CooperativeJob::bind(void* context,
                                                             JobDescriptor descriptor,
                                                             CooperativeJobOps operations) noexcept
{
  if (context == nullptr || !valid_job_descriptor(descriptor) || operations.begin == nullptr ||
      operations.advance == nullptr || operations.cancel == nullptr ||
      operations.state_bytes == nullptr || operations.save_state == nullptr ||
      operations.load_state == nullptr) {
    return std::unexpected(
        JobError{JobErrorCode::kInvalidDescriptor, "Cooperative job binding is incomplete"});
  }
  return CooperativeJob{context, descriptor, operations};
}

JobProgress CooperativeJob::progress() const noexcept
{
  return JobProgress{
      .state = state_,
      .completed_work_units = completed_work_units_,
      .remaining_work_units = descriptor_.total_work_units - completed_work_units_,
  };
}

std::size_t CooperativeJob::capsule_bytes() const noexcept
{
  return state_capsule_bytes(operations_.state_bytes(context_));
}

std::expected<JobProgress, JobError> CooperativeJob::start() noexcept
{
  if (state_ != JobState::kReady)
    return std::unexpected(
        JobError{JobErrorCode::kInvalidState, "Cooperative job has already started"});
  if (!operations_.begin(context_)) {
    state_ = JobState::kFailed;
    return std::unexpected(
        JobError{JobErrorCode::kCallbackFailed, "Cooperative job begin callback failed"});
  }
  state_ = JobState::kRunning;
  return progress();
}

std::expected<JobProgress, JobError> CooperativeJob::advance(
    std::size_t work_budget, std::uint64_t cancel_before_generation) noexcept
{
  if (terminal(state_))
    return progress();
  if (state_ != JobState::kRunning)
    return std::unexpected(
        JobError{JobErrorCode::kInvalidState, "Cooperative job must be started before advance"});
  if (cancel_before_generation > descriptor_.generation)
    return cancel_before(cancel_before_generation);
  if (work_budget == 0uz)
    return std::unexpected(
        JobError{JobErrorCode::kInvalidBudget, "Cooperative job work budget must be non-zero"});

  const std::uint64_t remaining = descriptor_.total_work_units - completed_work_units_;
  const std::uint64_t bounded = std::min<std::uint64_t>(remaining, work_budget);
  const std::size_t callback_budget = static_cast<std::size_t>(
      std::min<std::uint64_t>(bounded,
                              static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())));
  const BackendAdvance result = operations_.advance(context_, callback_budget);
  if (result.step != BackendStep::kInProgress && result.step != BackendStep::kComplete &&
      result.step != BackendStep::kFailed) {
    state_ = JobState::kFailed;
    return std::unexpected(JobError{JobErrorCode::kCallbackContract,
                                    "Cooperative job callback returned an invalid state"});
  }
  if (result.step == BackendStep::kFailed) {
    state_ = JobState::kFailed;
    return std::unexpected(
        JobError{JobErrorCode::kCallbackFailed, "Cooperative job advance callback failed"});
  }
  if (result.completed_work_units == 0uz || result.completed_work_units > callback_budget ||
      result.completed_work_units > remaining) {
    state_ = JobState::kFailed;
    return std::unexpected(JobError{JobErrorCode::kCallbackContract,
                                    "Cooperative job callback reported invalid progress"});
  }

  completed_work_units_ += result.completed_work_units;
  const bool exhausted = completed_work_units_ == descriptor_.total_work_units;
  if ((result.step == BackendStep::kComplete) != exhausted) {
    state_ = JobState::kFailed;
    return std::unexpected(JobError{JobErrorCode::kCallbackContract,
                                    "Cooperative job completion disagrees with its budget"});
  }
  state_ = exhausted ? JobState::kComplete : JobState::kRunning;
  return progress();
}

JobProgress CooperativeJob::cancel_before(std::uint64_t generation) noexcept
{
  if (!terminal(state_) && generation > descriptor_.generation) {
    operations_.cancel(context_);
    state_ = JobState::kCancelled;
  }
  return progress();
}

std::expected<std::size_t, JobError> CooperativeJob::export_capsule(
    std::span<std::byte> destination) const noexcept
{
  if (state_ == JobState::kReady)
    return std::unexpected(
        JobError{JobErrorCode::kInvalidState, "Cooperative job must start before state export"});
  const std::size_t payload_bytes = operations_.state_bytes(context_);
  const std::size_t required = state_capsule_bytes(payload_bytes);
  if (required == 0uz || destination.size() < required)
    return std::unexpected(
        JobError{JobErrorCode::kBufferTooSmall, "Cooperative job capsule buffer is too small"});
  auto payload = destination.subspan(kStateCapsuleHeaderBytes, payload_bytes);
  if (!operations_.save_state(context_, payload))
    return std::unexpected(
        JobError{JobErrorCode::kCallbackFailed, "Cooperative job state export failed"});
  const auto written = write_state_capsule(
      StateCapsuleMetadata{
          .descriptor = descriptor_,
          .state = state_,
          .completed_work_units = completed_work_units_,
      },
      payload, destination);
  if (!written)
    return std::unexpected(JobError{JobErrorCode::kCapsuleInvalid, written.error().message});
  return *written;
}

std::expected<JobProgress, JobError> CooperativeJob::restore_capsule(
    std::span<const std::byte> source) noexcept
{
  if (state_ != JobState::kReady)
    return std::unexpected(JobError{JobErrorCode::kInvalidState,
                                    "Only a fresh cooperative job can restore a capsule"});
  const auto capsule = read_state_capsule(source);
  if (!capsule)
    return std::unexpected(JobError{JobErrorCode::kCapsuleInvalid, capsule.error().message});
  if (capsule->metadata.state == JobState::kReady)
    return std::unexpected(
        JobError{JobErrorCode::kCapsuleInvalid, "A ready state capsule cannot resume execution"});
  if (!same_descriptor(descriptor_, capsule->metadata.descriptor))
    return std::unexpected(JobError{JobErrorCode::kCapsuleIncompatible,
                                    "State capsule belongs to an incompatible job"});
  if (!operations_.load_state(context_, capsule->payload))
    return std::unexpected(
        JobError{JobErrorCode::kCallbackFailed, "Cooperative job state import failed"});
  state_ = capsule->metadata.state;
  completed_work_units_ = capsule->metadata.completed_work_units;
  return progress();
}

} // namespace fe::relay
