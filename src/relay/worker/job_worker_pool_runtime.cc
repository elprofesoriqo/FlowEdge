#include "relay/worker/job_worker_pool_internal.h"

#include <limits>

namespace fe::relay {
using job_worker_pool_detail::make_failed_result;
using job_worker_pool_detail::monotonic_ns;

void JobWorkerPool::run_slot(Slot& slot) noexcept
{
  const WorkerBinding binding = bind_current_worker(slot.index, slot.placement);
  slot.numa_node.store(binding.numa_node, std::memory_order_relaxed);
  slot.logical_cpu.store(binding.logical_cpu, std::memory_order_relaxed);
  slot.bound.store(binding.bound, std::memory_order_release);

  while (!slot.stop_requested.load(std::memory_order_acquire)) {
    LaneState state = slot.state.load(std::memory_order_acquire);
    if (state != LaneState::kRequested) {
      slot.state.wait(state, std::memory_order_relaxed);
      continue;
    }

    const bool resuming = slot.resume_bytes != 0uz;
    slot.result.envelope = {};
    slot.result.metadata = {};
    slot.execution_failed = false;
    if (!resuming)
      slot.started_ns.store(monotonic_ns(), std::memory_order_relaxed);
    auto routed = slot.registry->bind(slot.request);
    if (!routed) {
      const JobResultCode code = routed.error().code == JobRouteErrorCode::kAdapterNotFound
                                     ? JobResultCode::kAdapterNotFound
                                     : JobResultCode::kFailed;
      make_failed_result(slot.result, slot.request, monotonic_ns(), code);
      slot.execution_failed = true;
    } else {
      slot.state.store(LaneState::kRunning, std::memory_order_release);
      bool handed_off{};
      auto progress =
          resuming
              ? routed->restore_capsule(std::span{slot.migration_storage}.first(slot.resume_bytes))
              : routed->start();
      slot.resume_bytes = 0uz;
      if (!progress) {
        slot.execution_failed = true;
      } else {
        if (resuming)
          slot.migration_completed_ns = monotonic_ns();
        const std::uint64_t cancel_generation =
            slot.cancel_generation.load(std::memory_order_acquire);
        if (cancel_generation > routed->descriptor().generation)
          progress = routed->cancel_before(cancel_generation);
      }

      const auto running = [&]() noexcept {
        return progress && progress->state == JobState::kRunning &&
               !slot.stop_requested.load(std::memory_order_acquire);
      };
      const auto advance_one = [&]() noexcept {
        progress = routed->advance(slot.work_quantum,
                                   slot.cancel_generation.load(std::memory_order_acquire));
        if (progress) {
          slot.remaining_work_units.store(progress->remaining_work_units,
                                          std::memory_order_release);
        } else {
          slot.execution_failed = true;
        }
      };
      if (slot.migration_storage.empty()) {
        while (running())
          advance_one();
      } else {
        while (running()) {
          if (!slot.migrate_requested.load(std::memory_order_acquire)) {
            advance_one();
            continue;
          }
          slot.migration_started_ns = monotonic_ns();
          const auto exported = routed->export_capsule(slot.migration_storage);
          if (exported) {
            slot.migration_bytes = *exported;
            slot.resume_completed_work_units = progress->completed_work_units;
            slot.remaining_work_units.store(progress->remaining_work_units,
                                            std::memory_order_relaxed);
            slot.state.store(LaneState::kMigrationReady, std::memory_order_release);
            slot.state.notify_one();
            handed_off = true;
            break;
          }
          slot.migrate_requested.store(false, std::memory_order_release);
          advance_one();
        }
      }

      if (slot.stop_requested.load(std::memory_order_acquire)) {
        static_cast<void>(routed->cancel_before(std::numeric_limits<std::uint64_t>::max()));
        break;
      }
      if (handed_off)
        continue;
      if (slot.execution_failed) {
        make_failed_result(slot.result, slot.request, monotonic_ns(), JobResultCode::kFailed);
      } else if (routed->write_result(slot.result, monotonic_ns()) != ProtocolResult::kSuccess) {
        make_failed_result(slot.result, slot.request, monotonic_ns(), JobResultCode::kFailed);
        slot.execution_failed = true;
      }
      slot.result.metadata.service_class = slot.request.metadata.service_class;
    }

    if (slot.stop_requested.load(std::memory_order_acquire))
      break;
    slot.remaining_work_units.store(0u, std::memory_order_release);
    slot.finished_ns.store(monotonic_ns(), std::memory_order_relaxed);
    slot.state.store(LaneState::kReady, std::memory_order_release);
    slot.state.notify_one();
  }
}

} // namespace fe::relay
