#pragma once

#include "relay/jobs/job_registry.h"
#include "relay/telemetry/job_event_buffer.h"
#include "relay/worker/worker_topology.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace fe::relay {

struct JobCostPolicy
{
  std::uint64_t iterative_ns{};
  std::uint64_t streaming_ns{};
  std::uint64_t speculative_ns{};
  std::uint64_t reserve_ns{};

  [[nodiscard]] constexpr std::uint64_t cost(JobKind kind) const noexcept
  {
    switch (kind) {
    case JobKind::kIterative:
      return iterative_ns;
    case JobKind::kStreaming:
      return streaming_ns;
    case JobKind::kSpeculative:
      return speculative_ns;
    case JobKind::kUnknown:
      break;
    }
    return 0u;
  }
};

struct JobQosPolicy
{
  std::size_t interactive_reserve_slots{};
  std::size_t critical_reserve_slots{};

  [[nodiscard]] constexpr std::size_t reserved_slots(JobServiceClass service_class) const noexcept
  {
    switch (service_class) {
    case JobServiceClass::kBestEffort:
      return interactive_reserve_slots + critical_reserve_slots;
    case JobServiceClass::kInteractive:
      return critical_reserve_slots;
    case JobServiceClass::kCritical:
      return 0uz;
    }
    return interactive_reserve_slots + critical_reserve_slots;
  }
};

struct WorkerSupervisionPolicy
{
  // Zero disables automatic quarantine.
  std::size_t consecutive_failure_threshold{};
};

enum class JobSubmitResult : std::uint8_t
{
  kAccepted,
  kStale,
  kDeadlineUnreachable,
  kInvalid,
  kAdapterNotFound,
  kFull,
  kQosCapacity,
};

enum class WorkerRecoveryResult : std::uint8_t
{
  kRecovered,
  kNotQuarantined,
  kNotDrained,
  kInvalidWorker,
};

enum class WorkerDrainResult : std::uint8_t
{
  kStarted,
  kAlreadyDraining,
  kWouldStrandWork,
  kInvalidWorker,
};

struct JobWorkerTiming
{
  std::uint64_t dispatched_ns{};
  std::uint64_t started_ns{};
  std::uint64_t finished_ns{};
};

// Bounded EDF execution for caller-owned cooperative backends. Each lane owns
// one frozen registry and therefore one mutable backend instance per route.
// Initialization allocates the slots and queue; submit, polling, cancellation,
// execution, migration, and result release do not allocate.
class JobWorkerPool
{
public:
  // Migration capacity is reserved per lane; zero keeps drain local to the lane.
  [[nodiscard]] static std::expected<JobWorkerPool, std::string> create(
      std::span<JobAdapterRegistry> lane_registries, std::size_t queue_capacity = 32uz,
      JobCostPolicy cost_policy = {}, std::size_t max_sessions = 64uz,
      std::size_t work_quantum = 1uz, WorkerPlacement placement = WorkerPlacement::kNone,
      JobEventBuffer* events = nullptr, std::size_t migration_capacity_bytes = 0uz,
      JobQosPolicy qos_policy = {}, WorkerSupervisionPolicy supervision_policy = {}) noexcept;

  ~JobWorkerPool();
  JobWorkerPool(const JobWorkerPool&) = delete;
  JobWorkerPool& operator=(const JobWorkerPool&) = delete;
  JobWorkerPool(JobWorkerPool&& other) noexcept;
  JobWorkerPool& operator=(JobWorkerPool&&) = delete;

  [[nodiscard]] std::size_t worker_count() const noexcept { return slots_.size(); }
  [[nodiscard]] std::size_t queue_capacity() const noexcept { return queue_storage_.size(); }
  [[nodiscard]] std::size_t queued_count() const noexcept { return queue_order_.size(); }
  [[nodiscard]] std::size_t busy_count() const noexcept;
  [[nodiscard]] bool has_idle() const noexcept;
  [[nodiscard]] std::uint64_t failure_count() const noexcept
  {
    return failure_count_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] std::uint64_t quarantine_count() const noexcept { return quarantine_count_; }
  [[nodiscard]] std::uint64_t qos_rejection_count() const noexcept { return qos_rejection_count_; }
  [[nodiscard]] WorkerBinding worker_binding(std::size_t index) const noexcept;
  [[nodiscard]] std::size_t migration_capacity_bytes() const noexcept
  {
    return migration_capacity_bytes_;
  }
  [[nodiscard]] std::size_t accepting_worker_count() const noexcept;
  [[nodiscard]] bool worker_busy(std::size_t index) const noexcept;
  [[nodiscard]] bool worker_draining(std::size_t index) const noexcept;
  [[nodiscard]] bool worker_drained(std::size_t index) const noexcept;
  [[nodiscard]] bool worker_quarantined(std::size_t index) const noexcept;
  [[nodiscard]] std::size_t worker_consecutive_failures(std::size_t index) const noexcept;
  // Drain and resume share the coordinator-thread ownership of submit/poll/release.
  [[nodiscard]] WorkerDrainResult request_worker_drain(std::size_t index) noexcept;
  [[nodiscard]] bool resume_worker(std::size_t index) noexcept;
  [[nodiscard]] WorkerRecoveryResult recover_worker(std::size_t index) noexcept;

  // A non-null rejection receives a typed result when the valid request is
  // rejected before enqueue. The coordinator calls submit/poll/release from one
  // thread; registered backends execute only on their owning worker lane.
  [[nodiscard]] JobSubmitResult submit(const JobRequestMessage& request, std::uint64_t now_ns = 0u,
                                       JobResultMessage* rejection = nullptr) noexcept;
  void cancel_before(std::uint64_t session_id, std::uint64_t generation) noexcept;
  // Releases an idle session watermark. Returns false while queued, running,
  // or unread results still reference the session.
  [[nodiscard]] bool release_session(std::uint64_t session_id) noexcept;

  // The returned view remains valid until release_ready_result().
  [[nodiscard]] const JobResultMessage* ready_result() noexcept;
  [[nodiscard]] JobWorkerTiming ready_timing() const noexcept;
  void release_ready_result() noexcept;

private:
  enum class LaneState : std::uint8_t;
  struct Slot;
  struct SessionWatermark
  {
    std::uint64_t session_id{};
    std::uint64_t generation{};
    bool occupied{};
  };

  JobWorkerPool() = default;
  static void run_slot(Slot& slot) noexcept;
  void stop() noexcept;
  void pump() noexcept;
  void handoff_migrations() noexcept;
  [[nodiscard]] bool dispatch_one(Slot& slot) noexcept;
  [[nodiscard]] bool admissible(const JobRequestMessage& request, std::uint64_t now_ns) noexcept;
  [[nodiscard]] bool supports_any(const JobRequestMessage& request) const noexcept;
  [[nodiscard]] bool has_accepting_worker(const JobRequestMessage& request) const noexcept;
  [[nodiscard]] bool has_qos_capacity(JobServiceClass service_class) const noexcept;
  [[nodiscard]] bool can_remove_worker(std::size_t index) const noexcept;
  [[nodiscard]] bool stale(const JobRequestMessage& request) const noexcept;
  [[nodiscard]] SessionWatermark* find_session(std::uint64_t session_id) noexcept;
  [[nodiscard]] const SessionWatermark* find_session(std::uint64_t session_id) const noexcept;
  [[nodiscard]] bool has_session_capacity(std::uint64_t session_id) const noexcept;
  void publish_rejection(JobResultMessage* destination, const JobRequestMessage& request,
                         std::uint64_t now_ns, JobResultCode code) noexcept;
  void emit(std::uint64_t sequence, const JobEventDetails& details) noexcept;
  void record_worker_failure(Slot& slot) noexcept;

  std::vector<std::unique_ptr<Slot>> slots_{};
  std::vector<JobRequestMessage> queue_storage_{};
  std::vector<std::size_t> queue_order_{};
  std::vector<std::size_t> free_queue_slots_{};
  std::vector<const JobRequestMessage*> admission_jobs_{};
  std::vector<SessionWatermark> sessions_{};
  JobCostPolicy cost_policy_{};
  JobQosPolicy qos_policy_{};
  WorkerSupervisionPolicy supervision_policy_{};
  std::size_t work_quantum_{1uz};
  std::optional<std::size_t> ready_index_{};
  JobEventBuffer* events_{};
  std::size_t dispatch_cursor_{};
  std::size_t ready_cursor_{};
  std::size_t migration_capacity_bytes_{};
  std::atomic<std::uint64_t> failure_count_{};
  std::uint64_t quarantine_count_{};
  std::uint64_t qos_rejection_count_{};
};

} // namespace fe::relay
