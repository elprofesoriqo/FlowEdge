#pragma once

#include "relay/jobs/job_types.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

namespace fe::relay {

enum class BackendStep : std::uint8_t
{
  kInProgress,
  kComplete,
  kFailed,
};

struct BackendAdvance
{
  BackendStep step{BackendStep::kFailed};
  std::size_t completed_work_units{};
};

struct CooperativeJobOps
{
  bool (*begin)(void* context) noexcept {};
  BackendAdvance (*advance)(void* context, std::size_t work_budget) noexcept {};
  void (*cancel)(void* context) noexcept {};
  std::size_t (*state_bytes)(const void* context) noexcept {};
  bool (*save_state)(const void* context, std::span<std::byte> destination) noexcept {};
  bool (*load_state)(void* context, std::span<const std::byte> source) noexcept {};
};

// Allocation-free, non-owning type erasure for a bounded stateful workload.
// The backend and every buffer it references must outlive this handle.
class CooperativeJob
{
public:
  [[nodiscard]] static std::expected<CooperativeJob, JobError> bind(
      void* context, JobDescriptor descriptor, CooperativeJobOps operations) noexcept;

  [[nodiscard]] const JobDescriptor& descriptor() const noexcept { return descriptor_; }
  [[nodiscard]] JobProgress progress() const noexcept;
  [[nodiscard]] std::size_t capsule_bytes() const noexcept;

  [[nodiscard]] std::expected<JobProgress, JobError> start() noexcept;
  [[nodiscard]] std::expected<JobProgress, JobError> advance(
      std::size_t work_budget, std::uint64_t cancel_before_generation = 0u) noexcept;
  [[nodiscard]] JobProgress cancel_before(std::uint64_t generation) noexcept;
  [[nodiscard]] std::expected<std::size_t, JobError> export_capsule(
      std::span<std::byte> destination) const noexcept;
  [[nodiscard]] std::expected<JobProgress, JobError> restore_capsule(
      std::span<const std::byte> source) noexcept;

private:
  CooperativeJob(void* context, JobDescriptor descriptor, CooperativeJobOps operations) noexcept
      : context_{context}, descriptor_{descriptor}, operations_{operations}
  {
  }

  void* context_{};
  JobDescriptor descriptor_{};
  CooperativeJobOps operations_{};
  JobState state_{JobState::kReady};
  std::uint64_t completed_work_units_{};
};

} // namespace fe::relay
