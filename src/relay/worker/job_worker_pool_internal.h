#pragma once

#include "relay/worker/job_worker_pool.h"

#include <chrono>
#include <limits>
#include <thread>
#include <utility>

namespace fe::relay {
namespace job_worker_pool_detail {

[[nodiscard]] inline std::uint64_t monotonic_ns() noexcept
{
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now().time_since_epoch())
                                        .count());
}

inline void make_failed_result(JobResultMessage& result, const JobRequestMessage& request,
                               std::uint64_t timestamp_ns, JobResultCode code) noexcept
{
  const ProtocolResult written =
      make_job_result(result, request.envelope.sequence, request.metadata.descriptor,
                      JobProgress{.state = JobState::kFailed,
                                  .completed_work_units = 0u,
                                  .remaining_work_units =
                                      request.metadata.descriptor.total_work_units},
                      timestamp_ns, code, {}, request.metadata.service_class);
  if (written != ProtocolResult::kSuccess)
    result = {};
}

} // namespace job_worker_pool_detail

enum class JobWorkerPool::LaneState : std::uint8_t
{
  kIdle,
  kRequested,
  kRunning,
  kMigrationReady,
  kReady,
  kDrained,
};

struct alignas(64) JobWorkerPool::Slot
{
  Slot(JobAdapterRegistry& source_registry, std::size_t source_index,
       std::size_t source_work_quantum, WorkerPlacement source_placement) noexcept
      : registry{&source_registry}, index{source_index}, work_quantum{source_work_quantum},
        placement{source_placement}
  {
  }

  JobAdapterRegistry* registry{};
  std::size_t index{};
  std::size_t work_quantum{1uz};
  WorkerPlacement placement{WorkerPlacement::kNone};
  JobRequestMessage request{};
  JobResultMessage result{};
  std::vector<std::byte> migration_storage{};
  std::atomic<LaneState> state{LaneState::kIdle};
  std::atomic_bool stop_requested{false};
  std::atomic_bool drain_requested{false};
  std::atomic_bool migrate_requested{false};
  std::atomic<std::uint64_t> session_id{};
  std::atomic<std::uint64_t> generation{};
  std::atomic<std::uint64_t> remaining_work_units{};
  std::atomic<std::uint64_t> cancel_generation{};
  std::atomic<std::uint64_t> preempted_ns{};
  std::atomic<std::uint64_t> dispatched_ns{};
  std::atomic<std::uint64_t> started_ns{};
  std::atomic<std::uint64_t> finished_ns{};
  std::atomic<std::uint32_t> numa_node{};
  std::atomic<std::uint32_t> logical_cpu{};
  std::atomic_bool bound{false};
  std::size_t migration_bytes{};
  std::size_t resume_bytes{};
  std::uint64_t resume_completed_work_units{};
  std::uint64_t migration_started_ns{};
  std::uint64_t migration_completed_ns{};
  std::uint32_t migration_source{kNoWorker};
  bool execution_failed{};
  bool events_published{};
  std::size_t consecutive_failures{};
  bool quarantined{};
  std::thread thread{};
};

} // namespace fe::relay
