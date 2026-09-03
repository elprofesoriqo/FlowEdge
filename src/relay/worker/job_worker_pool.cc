#include "relay/worker/job_worker_pool.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <ranges>
#include <thread>
#include <utility>

namespace fe::relay {
namespace {

[[nodiscard]] std::uint64_t monotonic_ns() noexcept
{
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now().time_since_epoch())
                                        .count());
}

[[nodiscard]] constexpr std::uint64_t deadline_key(const JobRequestMessage& message) noexcept
{
  return message.metadata.descriptor.deadline_ns == 0u ? std::numeric_limits<std::uint64_t>::max()
                                                       : message.metadata.descriptor.deadline_ns;
}

[[nodiscard]] constexpr bool earlier(const JobRequestMessage& left,
                                     const JobRequestMessage& right) noexcept
{
  const std::uint64_t left_deadline = deadline_key(left);
  const std::uint64_t right_deadline = deadline_key(right);
  if (left_deadline != right_deadline)
    return left_deadline < right_deadline;
  if (left.metadata.service_class != right.metadata.service_class)
    return left.metadata.service_class > right.metadata.service_class;
  if (left.metadata.timestamp_ns != right.metadata.timestamp_ns)
    return left.metadata.timestamp_ns < right.metadata.timestamp_ns;
  return left.envelope.sequence < right.envelope.sequence;
}

[[nodiscard]] constexpr std::uint64_t saturating_add(std::uint64_t left,
                                                     std::uint64_t right) noexcept
{
  return left > std::numeric_limits<std::uint64_t>::max() - right
             ? std::numeric_limits<std::uint64_t>::max()
             : left + right;
}

[[nodiscard]] constexpr std::uint64_t saturating_multiply(std::uint64_t left,
                                                          std::uint64_t right) noexcept
{
  return right != 0u && left > std::numeric_limits<std::uint64_t>::max() / right
             ? std::numeric_limits<std::uint64_t>::max()
             : left * right;
}

[[nodiscard]] constexpr std::uint64_t elapsed(std::uint64_t start, std::uint64_t end) noexcept
{
  return start != 0u && end >= start ? end - start : 0u;
}

void update_max(std::atomic<std::uint64_t>& value, std::uint64_t candidate) noexcept
{
  std::uint64_t current = value.load(std::memory_order_relaxed);
  while (current < candidate &&
         !value.compare_exchange_weak(current, candidate, std::memory_order_release,
                                      std::memory_order_relaxed)) {
  }
}

[[nodiscard]] bool registration_accepts(const JobAdapterRegistry& registry,
                                        const JobRequestMessage& request) noexcept
{
  const JobAdapterRegistration* const registration = registry.find(request.metadata.descriptor);
  return registration != nullptr &&
         request.metadata.payload_bytes <= registration->max_request_bytes;
}

void make_failed_result(JobResultMessage& result, const JobRequestMessage& request,
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

} // namespace

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

std::expected<JobWorkerPool, std::string> JobWorkerPool::create(
    std::span<JobAdapterRegistry> lane_registries, std::size_t queue_capacity,
    JobCostPolicy cost_policy, std::size_t max_sessions, std::size_t work_quantum,
    WorkerPlacement placement, JobEventBuffer* events, std::size_t migration_capacity_bytes,
    JobQosPolicy qos_policy, WorkerSupervisionPolicy supervision_policy) noexcept
{
  if (lane_registries.empty() || lane_registries.size() > 8uz)
    return std::unexpected("Generic Relay worker count must be in the range 1..8");
  if (queue_capacity == 0uz || max_sessions == 0uz || work_quantum == 0uz)
    return std::unexpected("Generic Relay queue, session, and work limits must be non-zero");
  if (qos_policy.interactive_reserve_slots >= queue_capacity ||
      qos_policy.critical_reserve_slots >= queue_capacity ||
      qos_policy.interactive_reserve_slots >
          queue_capacity - 1uz - qos_policy.critical_reserve_slots)
    return std::unexpected("Generic Relay QoS reserves must leave one best-effort queue slot");
  for (const JobAdapterRegistry& registry : lane_registries)
    if (!registry.frozen() || registry.size() == 0uz)
      return std::unexpected("Every generic Relay worker lane requires a frozen adapter registry");

  try {
    JobWorkerPool pool{};
    pool.cost_policy_ = cost_policy;
    pool.qos_policy_ = qos_policy;
    pool.supervision_policy_ = supervision_policy;
    pool.work_quantum_ = work_quantum;
    pool.events_ = events;
    pool.migration_capacity_bytes_ = migration_capacity_bytes;
    pool.queue_storage_.resize(queue_capacity);
    pool.queue_order_.reserve(queue_capacity);
    pool.free_queue_slots_.reserve(queue_capacity);
    pool.admission_jobs_.reserve(queue_capacity + 1uz);
    pool.sessions_.resize(max_sessions);
    for (std::size_t slot{queue_capacity}; slot > 0uz; --slot)
      pool.free_queue_slots_.push_back(slot - 1uz);

    pool.slots_.reserve(lane_registries.size());
    for (std::size_t index{}; index < lane_registries.size(); ++index) {
      pool.slots_.push_back(
          std::make_unique<Slot>(lane_registries[index], index, work_quantum, placement));
      pool.slots_.back()->migration_storage.resize(migration_capacity_bytes);
    }
    for (const auto& slot : pool.slots_)
      slot->thread = std::thread{JobWorkerPool::run_slot, std::ref(*slot)};
    return pool;
  } catch (const std::exception& error) {
    return std::unexpected("Failed to create generic Relay worker pool: " +
                           std::string{error.what()});
  } catch (...) {
    return std::unexpected("Failed to create generic Relay worker pool");
  }
}

JobWorkerPool::~JobWorkerPool()
{
  stop();
}

JobWorkerPool::JobWorkerPool(JobWorkerPool&& other) noexcept
    : slots_{std::move(other.slots_)}, queue_storage_{std::move(other.queue_storage_)},
      queue_order_{std::move(other.queue_order_)},
      free_queue_slots_{std::move(other.free_queue_slots_)},
      admission_jobs_{std::move(other.admission_jobs_)}, sessions_{std::move(other.sessions_)},
      cost_policy_{other.cost_policy_}, qos_policy_{other.qos_policy_},
      supervision_policy_{other.supervision_policy_}, work_quantum_{other.work_quantum_},
      ready_index_{other.ready_index_}, events_{other.events_},
      dispatch_cursor_{other.dispatch_cursor_}, ready_cursor_{other.ready_cursor_},
      migration_capacity_bytes_{other.migration_capacity_bytes_},
      failure_count_{other.failure_count_.load(std::memory_order_relaxed)},
      quarantine_count_{other.quarantine_count_}, qos_rejection_count_{other.qos_rejection_count_}
{
  other.ready_index_.reset();
}

std::size_t JobWorkerPool::busy_count() const noexcept
{
  return static_cast<std::size_t>(std::ranges::count_if(slots_, [](const auto& slot) {
    const LaneState state = slot->state.load(std::memory_order_acquire);
    return state == LaneState::kRequested || state == LaneState::kRunning ||
           state == LaneState::kMigrationReady;
  }));
}

bool JobWorkerPool::has_idle() const noexcept
{
  return std::ranges::any_of(slots_, [](const auto& slot) {
    return !slot->drain_requested.load(std::memory_order_acquire) &&
           slot->state.load(std::memory_order_acquire) == LaneState::kIdle;
  });
}

WorkerBinding JobWorkerPool::worker_binding(std::size_t index) const noexcept
{
  if (index >= slots_.size())
    return {};
  const Slot& slot = *slots_[index];
  return WorkerBinding{.numa_node = slot.numa_node.load(std::memory_order_relaxed),
                       .logical_cpu = slot.logical_cpu.load(std::memory_order_relaxed),
                       .bound = slot.bound.load(std::memory_order_acquire)};
}

std::size_t JobWorkerPool::accepting_worker_count() const noexcept
{
  return static_cast<std::size_t>(std::ranges::count_if(slots_, [](const auto& slot) {
    return !slot->drain_requested.load(std::memory_order_acquire);
  }));
}

bool JobWorkerPool::worker_busy(std::size_t index) const noexcept
{
  if (index >= slots_.size())
    return false;
  const LaneState state = slots_[index]->state.load(std::memory_order_acquire);
  return state == LaneState::kRequested || state == LaneState::kRunning ||
         state == LaneState::kMigrationReady;
}

bool JobWorkerPool::worker_draining(std::size_t index) const noexcept
{
  return index < slots_.size() && slots_[index]->drain_requested.load(std::memory_order_acquire);
}

bool JobWorkerPool::worker_drained(std::size_t index) const noexcept
{
  return index < slots_.size() &&
         slots_[index]->state.load(std::memory_order_acquire) == LaneState::kDrained;
}

bool JobWorkerPool::worker_quarantined(std::size_t index) const noexcept
{
  return index < slots_.size() && slots_[index]->quarantined;
}

std::size_t JobWorkerPool::worker_consecutive_failures(std::size_t index) const noexcept
{
  return index < slots_.size() ? slots_[index]->consecutive_failures : 0uz;
}

bool JobWorkerPool::can_remove_worker(std::size_t index) const noexcept
{
  const Slot& selected = *slots_[index];
  for (const std::size_t queue_slot : queue_order_) {
    const JobRequestMessage& queued = queue_storage_[queue_slot];
    if (!registration_accepts(*selected.registry, queued))
      continue;
    const bool has_target = std::ranges::any_of(slots_, [&](const auto& candidate) {
      return candidate->index != index && !candidate->quarantined &&
             !candidate->drain_requested.load(std::memory_order_acquire) &&
             registration_accepts(*candidate->registry, queued);
    });
    if (!has_target)
      return false;
  }
  return true;
}

WorkerDrainResult JobWorkerPool::request_worker_drain(std::size_t index) noexcept
{
  if (index >= slots_.size())
    return WorkerDrainResult::kInvalidWorker;
  Slot& slot = *slots_[index];
  if (slot.drain_requested.load(std::memory_order_acquire))
    return WorkerDrainResult::kAlreadyDraining;
  if (!can_remove_worker(index))
    return WorkerDrainResult::kWouldStrandWork;
  slot.drain_requested.store(true, std::memory_order_release);

  LaneState state = slot.state.load(std::memory_order_acquire);
  if (state == LaneState::kIdle &&
      slot.state.compare_exchange_strong(state, LaneState::kDrained, std::memory_order_acq_rel,
                                         std::memory_order_acquire)) {
    slot.session_id.store(0u, std::memory_order_relaxed);
    slot.generation.store(0u, std::memory_order_relaxed);
    slot.state.notify_one();
    return WorkerDrainResult::kStarted;
  }

  if ((state == LaneState::kRequested || state == LaneState::kRunning) &&
      slot.migration_source == kNoWorker && migration_capacity_bytes_ != 0uz) {
    const bool has_target = std::ranges::any_of(slots_, [&](const auto& candidate) {
      return candidate->index != index &&
             !candidate->drain_requested.load(std::memory_order_acquire) &&
             registration_accepts(*candidate->registry, slot.request);
    });
    slot.migrate_requested.store(has_target, std::memory_order_release);
  }
  pump();
  return WorkerDrainResult::kStarted;
}

bool JobWorkerPool::resume_worker(std::size_t index) noexcept
{
  if (index >= slots_.size())
    return false;
  Slot& slot = *slots_[index];
  if (slot.quarantined || slot.state.load(std::memory_order_acquire) != LaneState::kDrained)
    return false;
  slot.migrate_requested.store(false, std::memory_order_relaxed);
  slot.drain_requested.store(false, std::memory_order_release);
  slot.state.store(LaneState::kIdle, std::memory_order_release);
  slot.state.notify_one();
  pump();
  return true;
}

WorkerRecoveryResult JobWorkerPool::recover_worker(std::size_t index) noexcept
{
  if (index >= slots_.size())
    return WorkerRecoveryResult::kInvalidWorker;
  Slot& slot = *slots_[index];
  if (!slot.quarantined)
    return WorkerRecoveryResult::kNotQuarantined;
  if (slot.state.load(std::memory_order_acquire) != LaneState::kDrained)
    return WorkerRecoveryResult::kNotDrained;
  slot.consecutive_failures = 0uz;
  slot.quarantined = false;
  slot.migrate_requested.store(false, std::memory_order_relaxed);
  slot.drain_requested.store(false, std::memory_order_release);
  slot.state.store(LaneState::kIdle, std::memory_order_release);
  slot.state.notify_one();
  pump();
  return WorkerRecoveryResult::kRecovered;
}

JobWorkerPool::SessionWatermark* JobWorkerPool::find_session(std::uint64_t session_id) noexcept
{
  const auto found = std::ranges::find_if(sessions_, [session_id](const SessionWatermark& entry) {
    return entry.occupied && entry.session_id == session_id;
  });
  return found == sessions_.end() ? nullptr : &*found;
}

const JobWorkerPool::SessionWatermark* JobWorkerPool::find_session(
    std::uint64_t session_id) const noexcept
{
  const auto found = std::ranges::find_if(sessions_, [session_id](const SessionWatermark& entry) {
    return entry.occupied && entry.session_id == session_id;
  });
  return found == sessions_.end() ? nullptr : &*found;
}

bool JobWorkerPool::has_session_capacity(std::uint64_t session_id) const noexcept
{
  return find_session(session_id) != nullptr ||
         std::ranges::any_of(sessions_,
                             [](const SessionWatermark& entry) { return !entry.occupied; });
}

bool JobWorkerPool::stale(const JobRequestMessage& request) const noexcept
{
  const SessionWatermark* const watermark = find_session(request.metadata.descriptor.session_id);
  return watermark != nullptr && request.metadata.descriptor.generation < watermark->generation;
}

bool JobWorkerPool::supports_any(const JobRequestMessage& request) const noexcept
{
  return std::ranges::any_of(slots_, [&](const auto& slot) {
    return registration_accepts(*slot->registry, request);
  });
}

bool JobWorkerPool::has_accepting_worker(const JobRequestMessage& request) const noexcept
{
  return std::ranges::any_of(slots_, [&](const auto& slot) {
    return !slot->drain_requested.load(std::memory_order_acquire) &&
           registration_accepts(*slot->registry, request);
  });
}

bool JobWorkerPool::has_qos_capacity(JobServiceClass service_class) const noexcept
{
  const std::size_t available = queue_storage_.size() - queue_order_.size();
  return available > qos_policy_.reserved_slots(service_class);
}

bool JobWorkerPool::admissible(const JobRequestMessage& request, std::uint64_t now_ns) noexcept
{
  const JobDescriptor& candidate = request.metadata.descriptor;
  const std::uint64_t candidate_cost = cost_policy_.cost(candidate.kind);
  if (candidate_cost == 0u || now_ns == 0u || candidate.deadline_ns == 0u)
    return true;
  if (candidate.deadline_ns <= now_ns)
    return false;

  std::array<std::uint64_t, 8> available_at{};
  std::ranges::fill(available_at, now_ns);
  for (std::size_t index{}; index < slots_.size(); ++index) {
    const Slot& slot = *slots_[index];
    if (slot.drain_requested.load(std::memory_order_acquire)) {
      available_at[index] = std::numeric_limits<std::uint64_t>::max();
      continue;
    }
    const LaneState state = slot.state.load(std::memory_order_acquire);
    if (state == LaneState::kIdle)
      continue;
    if (state == LaneState::kReady) {
      available_at[index] = std::numeric_limits<std::uint64_t>::max();
      continue;
    }
    const JobDescriptor& active = slot.request.metadata.descriptor;
    const std::uint64_t active_cost = cost_policy_.cost(active.kind);
    if (active_cost == 0u)
      return true;
    std::uint64_t remaining = slot.remaining_work_units.load(std::memory_order_acquire);
    if (active.session_id == candidate.session_id && active.generation < candidate.generation)
      remaining = std::min<std::uint64_t>(remaining, work_quantum_);
    available_at[index] = saturating_add(now_ns, saturating_multiply(remaining, active_cost));
  }

  admission_jobs_.clear();
  admission_jobs_.push_back(&request);
  for (const std::size_t slot : queue_order_) {
    const JobRequestMessage& queued = queue_storage_[slot];
    const JobDescriptor& descriptor = queued.metadata.descriptor;
    if (stale(queued) || descriptor.deadline_ns == 0u ||
        (descriptor.session_id == candidate.session_id &&
         descriptor.generation < candidate.generation))
      continue;
    admission_jobs_.push_back(&queued);
  }
  std::ranges::sort(admission_jobs_,
                    [](const JobRequestMessage* left, const JobRequestMessage* right) {
                      return earlier(*left, *right);
                    });

  for (const JobRequestMessage* const job : admission_jobs_) {
    const JobDescriptor& descriptor = job->metadata.descriptor;
    const std::uint64_t unit_cost = cost_policy_.cost(descriptor.kind);
    if (unit_cost == 0u)
      return true;
    std::optional<std::size_t> best_lane{};
    for (std::size_t lane{}; lane < slots_.size(); ++lane) {
      if (available_at[lane] == std::numeric_limits<std::uint64_t>::max() ||
          !registration_accepts(*slots_[lane]->registry, *job))
        continue;
      if (!best_lane || available_at[lane] < available_at[*best_lane])
        best_lane = lane;
    }
    if (!best_lane)
      return false;
    const std::uint64_t duration = saturating_multiply(descriptor.total_work_units, unit_cost);
    const std::uint64_t start = available_at[*best_lane];
    if (start > descriptor.deadline_ns || duration > descriptor.deadline_ns - start)
      return false;
    const std::uint64_t finish = start + duration;
    if (cost_policy_.reserve_ns > descriptor.deadline_ns - finish)
      return false;
    available_at[*best_lane] = finish;
  }
  return true;
}

void JobWorkerPool::publish_rejection(JobResultMessage* destination,
                                      const JobRequestMessage& request, std::uint64_t now_ns,
                                      JobResultCode code) noexcept
{
  if (destination != nullptr) {
    destination->envelope = {};
    destination->metadata = {};
    static_cast<void>(make_rejected_job_result(*destination, request, now_ns, code));
  }
  emit(request.envelope.sequence,
       JobEventDetails{.event = JobEventKind::kRejected,
                       .descriptor = request.metadata.descriptor,
                       .progress = JobProgress{.state = JobState::kFailed,
                                               .completed_work_units = 0u,
                                               .remaining_work_units =
                                                   request.metadata.descriptor.total_work_units},
                       .timestamp_ns = now_ns == 0u ? monotonic_ns() : now_ns,
                       .result_code = code,
                       .service_class = request.metadata.service_class});
}

void JobWorkerPool::emit(std::uint64_t sequence, const JobEventDetails& details) noexcept
{
  if (events_ == nullptr)
    return;
  JobEventMessage event{};
  if (make_job_event(event, sequence, details) == ProtocolResult::kSuccess)
    static_cast<void>(events_->try_push(event));
}

void JobWorkerPool::record_worker_failure(Slot& slot) noexcept
{
  failure_count_.fetch_add(1u, std::memory_order_relaxed);
  ++slot.consecutive_failures;
  if (slot.quarantined || supervision_policy_.consecutive_failure_threshold == 0uz ||
      slot.consecutive_failures < supervision_policy_.consecutive_failure_threshold ||
      !can_remove_worker(slot.index))
    return;
  slot.quarantined = true;
  ++quarantine_count_;
  slot.drain_requested.store(true, std::memory_order_release);
}

JobSubmitResult JobWorkerPool::submit(const JobRequestMessage& request, std::uint64_t now_ns,
                                      JobResultMessage* rejection) noexcept
{
  if (validate(request) != ProtocolResult::kSuccess) {
    publish_rejection(rejection, request, now_ns, JobResultCode::kInvalidRequest);
    return JobSubmitResult::kInvalid;
  }
  if (!supports_any(request)) {
    publish_rejection(rejection, request, now_ns, JobResultCode::kAdapterNotFound);
    return JobSubmitResult::kAdapterNotFound;
  }
  if (!has_accepting_worker(request)) {
    publish_rejection(rejection, request, now_ns, JobResultCode::kRejectedCapacity);
    return JobSubmitResult::kFull;
  }
  if (stale(request)) {
    publish_rejection(rejection, request, now_ns, JobResultCode::kRejectedStale);
    return JobSubmitResult::kStale;
  }
  if (!has_session_capacity(request.metadata.descriptor.session_id) ||
      queue_order_.size() == queue_storage_.size()) {
    publish_rejection(rejection, request, now_ns, JobResultCode::kRejectedCapacity);
    return JobSubmitResult::kFull;
  }
  if (!has_qos_capacity(request.metadata.service_class)) {
    ++qos_rejection_count_;
    publish_rejection(rejection, request, now_ns, JobResultCode::kRejectedQos);
    return JobSubmitResult::kQosCapacity;
  }
  if (!admissible(request, now_ns)) {
    publish_rejection(rejection, request, now_ns, JobResultCode::kRejectedDeadline);
    return JobSubmitResult::kDeadlineUnreachable;
  }

  const std::size_t queue_slot = free_queue_slots_.back();
  free_queue_slots_.pop_back();
  std::memcpy(&queue_storage_[queue_slot], &request, wire_size(request));
  queue_order_.push_back(queue_slot);

  emit(request.envelope.sequence,
       JobEventDetails{.event = JobEventKind::kAdmitted,
                       .descriptor = request.metadata.descriptor,
                       .progress = JobProgress{.state = JobState::kReady,
                                               .completed_work_units = 0u,
                                               .remaining_work_units =
                                                   request.metadata.descriptor.total_work_units},
                       .timestamp_ns = now_ns == 0u ? monotonic_ns() : now_ns,
                       .queue_depth = static_cast<std::uint32_t>(queue_order_.size()),
                       .service_class = request.metadata.service_class});
  SessionWatermark* watermark = find_session(request.metadata.descriptor.session_id);
  if (watermark == nullptr) {
    watermark = &*std::ranges::find_if(sessions_, [](const SessionWatermark& entry) {
      return !entry.occupied;
    });
    watermark->occupied = true;
    watermark->session_id = request.metadata.descriptor.session_id;
    watermark->generation = request.metadata.descriptor.generation;
  } else if (request.metadata.descriptor.generation > watermark->generation) {
    watermark->generation = request.metadata.descriptor.generation;
    cancel_before(watermark->session_id, watermark->generation);
  }
  pump();
  return JobSubmitResult::kAccepted;
}

void JobWorkerPool::cancel_before(std::uint64_t session_id, std::uint64_t generation) noexcept
{
  SessionWatermark* watermark = find_session(session_id);
  if (watermark == nullptr) {
    const auto free = std::ranges::find_if(sessions_, [](const SessionWatermark& entry) {
      return !entry.occupied;
    });
    if (free == sessions_.end())
      return;
    free->occupied = true;
    free->session_id = session_id;
    free->generation = generation;
    watermark = &*free;
  } else {
    watermark->generation = std::max(watermark->generation, generation);
  }

  const std::uint64_t now_ns = monotonic_ns();
  for (std::size_t index{}; index < slots_.size(); ++index) {
    Slot& slot = *slots_[index];
    if (slot.session_id.load(std::memory_order_acquire) != session_id ||
        slot.generation.load(std::memory_order_acquire) >= watermark->generation)
      continue;
    const LaneState state = slot.state.load(std::memory_order_acquire);
    if (state == LaneState::kRequested || state == LaneState::kRunning ||
        state == LaneState::kMigrationReady) {
      update_max(slot.cancel_generation, watermark->generation);
      slot.preempted_ns.store(now_ns, std::memory_order_release);
      const std::uint64_t remaining = slot.remaining_work_units.load(std::memory_order_acquire);
      emit(slot.request.envelope.sequence,
           JobEventDetails{.event = JobEventKind::kPreempted,
                           .descriptor = slot.request.metadata.descriptor,
                           .progress =
                               JobProgress{.state = JobState::kRunning,
                                           .completed_work_units =
                                               slot.request.metadata.descriptor.total_work_units -
                                               remaining,
                                           .remaining_work_units = remaining},
                           .timestamp_ns = now_ns,
                           .related_generation = watermark->generation,
                           .worker_index = static_cast<std::uint32_t>(index),
                           .service_class = slot.request.metadata.service_class});
    } else if (state == LaneState::kReady) {
      static_cast<void>(make_rejected_job_result(slot.result, slot.request, now_ns,
                                                 JobResultCode::kRejectedStale));
    }
  }
  pump();
}

bool JobWorkerPool::release_session(std::uint64_t session_id) noexcept
{
  if (find_session(session_id) == nullptr)
    return true;
  if (std::ranges::any_of(queue_order_, [&](std::size_t slot) {
        return queue_storage_[slot].metadata.descriptor.session_id == session_id;
      }))
    return false;
  if (std::ranges::any_of(slots_, [&](const auto& slot) {
        return slot->state.load(std::memory_order_acquire) != LaneState::kIdle &&
               slot->session_id.load(std::memory_order_acquire) == session_id;
      }))
    return false;
  *find_session(session_id) = {};
  return true;
}

bool JobWorkerPool::dispatch_one(Slot& slot) noexcept
{
  if (slot.drain_requested.load(std::memory_order_acquire))
    return false;
  const std::uint64_t dispatched_ns = monotonic_ns();
  std::optional<std::size_t> selected_position{};
  for (std::size_t position{}; position < queue_order_.size(); ++position) {
    const JobRequestMessage& request = queue_storage_[queue_order_[position]];
    const bool rejected =
        stale(request) || (request.metadata.descriptor.deadline_ns != 0u &&
                           request.metadata.descriptor.deadline_ns <= dispatched_ns);
    if (!rejected && !registration_accepts(*slot.registry, request))
      continue;
    if (!selected_position || earlier(request, queue_storage_[queue_order_[*selected_position]]))
      selected_position = position;
  }
  if (!selected_position)
    return false;

  const std::size_t queue_slot = queue_order_[*selected_position];
  const JobRequestMessage& request = queue_storage_[queue_slot];
  std::memcpy(&slot.request, &request, wire_size(request));
  queue_order_.erase(queue_order_.begin() + static_cast<std::ptrdiff_t>(*selected_position));
  free_queue_slots_.push_back(queue_slot);

  slot.session_id.store(request.metadata.descriptor.session_id, std::memory_order_relaxed);
  slot.generation.store(request.metadata.descriptor.generation, std::memory_order_relaxed);
  slot.remaining_work_units.store(request.metadata.descriptor.total_work_units,
                                  std::memory_order_relaxed);
  slot.cancel_generation.store(0u, std::memory_order_relaxed);
  slot.preempted_ns.store(0u, std::memory_order_relaxed);
  slot.dispatched_ns.store(dispatched_ns, std::memory_order_relaxed);
  slot.started_ns.store(0u, std::memory_order_relaxed);
  slot.finished_ns.store(0u, std::memory_order_relaxed);
  slot.migration_bytes = 0uz;
  slot.resume_bytes = 0uz;
  slot.resume_completed_work_units = 0u;
  slot.migration_started_ns = 0u;
  slot.migration_completed_ns = 0u;
  slot.migration_source = kNoWorker;
  slot.migrate_requested.store(false, std::memory_order_relaxed);
  slot.events_published = false;

  emit(slot.request.envelope.sequence,
       JobEventDetails{.event = JobEventKind::kDispatched,
                       .descriptor = slot.request.metadata.descriptor,
                       .progress =
                           JobProgress{.state = JobState::kReady,
                                       .completed_work_units = 0u,
                                       .remaining_work_units =
                                           slot.request.metadata.descriptor.total_work_units},
                       .timestamp_ns = dispatched_ns,
                       .worker_index = static_cast<std::uint32_t>(slot.index),
                       .queue_depth = static_cast<std::uint32_t>(queue_order_.size()),
                       .service_class = slot.request.metadata.service_class});

  if (stale(slot.request)) {
    static_cast<void>(make_rejected_job_result(slot.result, slot.request, dispatched_ns,
                                               JobResultCode::kRejectedStale));
    slot.started_ns.store(dispatched_ns, std::memory_order_relaxed);
    slot.finished_ns.store(dispatched_ns, std::memory_order_relaxed);
    slot.state.store(LaneState::kReady, std::memory_order_release);
  } else if (slot.request.metadata.descriptor.deadline_ns != 0u &&
             slot.request.metadata.descriptor.deadline_ns <= dispatched_ns) {
    static_cast<void>(make_rejected_job_result(slot.result, slot.request, dispatched_ns,
                                               JobResultCode::kRejectedDeadline));
    slot.started_ns.store(dispatched_ns, std::memory_order_relaxed);
    slot.finished_ns.store(dispatched_ns, std::memory_order_relaxed);
    slot.state.store(LaneState::kReady, std::memory_order_release);
  } else {
    slot.state.store(LaneState::kRequested, std::memory_order_release);
    slot.state.notify_one();
  }
  return true;
}

void JobWorkerPool::pump() noexcept
{
  if (migration_capacity_bytes_ != 0uz)
    handoff_migrations();
  if (queue_order_.empty())
    return;
  for (std::size_t checked{}; checked < slots_.size() && !queue_order_.empty(); ++checked) {
    const std::size_t index = (dispatch_cursor_ + checked) % slots_.size();
    Slot& slot = *slots_[index];
    if (slot.drain_requested.load(std::memory_order_acquire) ||
        slot.state.load(std::memory_order_acquire) != LaneState::kIdle)
      continue;
    if (dispatch_one(slot))
      dispatch_cursor_ = (index + 1uz) % slots_.size();
  }
}

void JobWorkerPool::handoff_migrations() noexcept
{
  for (const auto& source_pointer : slots_) {
    Slot& source = *source_pointer;
    if (source.state.load(std::memory_order_acquire) != LaneState::kMigrationReady)
      continue;

    Slot* destination{};
    bool has_target{};
    for (const auto& candidate_pointer : slots_) {
      Slot& candidate = *candidate_pointer;
      if (candidate.index == source.index ||
          candidate.drain_requested.load(std::memory_order_acquire) ||
          !registration_accepts(*candidate.registry, source.request) ||
          candidate.migration_storage.size() < source.migration_bytes)
        continue;
      has_target = true;
      if (candidate.state.load(std::memory_order_acquire) == LaneState::kIdle) {
        destination = &candidate;
        break;
      }
    }

    if (destination == nullptr) {
      if (has_target)
        continue;
      source.resume_bytes = source.migration_bytes;
      source.migration_bytes = 0uz;
      source.migrate_requested.store(false, std::memory_order_relaxed);
      source.migration_started_ns = 0u;
      source.state.store(LaneState::kRequested, std::memory_order_release);
      source.state.notify_one();
      continue;
    }

    std::ranges::copy(std::span{source.migration_storage}.first(source.migration_bytes),
                      destination->migration_storage.begin());
    std::memcpy(&destination->request, &source.request, wire_size(source.request));
    destination->session_id.store(source.session_id.load(std::memory_order_relaxed),
                                  std::memory_order_relaxed);
    destination->generation.store(source.generation.load(std::memory_order_relaxed),
                                  std::memory_order_relaxed);
    destination->remaining_work_units.store(source.remaining_work_units.load(
                                                std::memory_order_relaxed),
                                            std::memory_order_relaxed);
    destination->cancel_generation.store(source.cancel_generation.load(std::memory_order_relaxed),
                                         std::memory_order_relaxed);
    destination->preempted_ns.store(source.preempted_ns.load(std::memory_order_relaxed),
                                    std::memory_order_relaxed);
    destination->dispatched_ns.store(source.dispatched_ns.load(std::memory_order_relaxed),
                                     std::memory_order_relaxed);
    destination->started_ns.store(source.started_ns.load(std::memory_order_relaxed),
                                  std::memory_order_relaxed);
    destination->finished_ns.store(0u, std::memory_order_relaxed);
    destination->resume_bytes = source.migration_bytes;
    destination->resume_completed_work_units = source.resume_completed_work_units;
    destination->migration_started_ns = source.migration_started_ns;
    destination->migration_completed_ns = 0u;
    destination->migration_source = static_cast<std::uint32_t>(source.index);
    destination->migration_bytes = 0uz;
    destination->events_published = false;
    destination->execution_failed = false;

    source.migration_bytes = 0uz;
    source.remaining_work_units.store(0u, std::memory_order_relaxed);
    source.session_id.store(0u, std::memory_order_relaxed);
    source.generation.store(0u, std::memory_order_relaxed);
    source.state.store(LaneState::kDrained, std::memory_order_release);
    source.state.notify_one();
    destination->state.store(LaneState::kRequested, std::memory_order_release);
    destination->state.notify_one();
  }
}

const JobResultMessage* JobWorkerPool::ready_result() noexcept
{
  if (migration_capacity_bytes_ != 0uz)
    handoff_migrations();
  if (ready_index_) {
    if (slots_[*ready_index_]->state.load(std::memory_order_acquire) == LaneState::kReady)
      return &slots_[*ready_index_]->result;
    ready_index_.reset();
  }
  for (std::size_t checked{}; checked < slots_.size(); ++checked) {
    const std::size_t index = (ready_cursor_ + checked) % slots_.size();
    Slot& slot = *slots_[index];
    if (slot.state.load(std::memory_order_acquire) != LaneState::kReady)
      continue;
    const bool execution_failed = std::exchange(slot.execution_failed, false);
    const bool valid_result = validate(slot.result) == ProtocolResult::kSuccess;
    if (execution_failed || !valid_result)
      record_worker_failure(slot);
    if (!valid_result) {
      slot.migration_source = kNoWorker;
      slot.resume_completed_work_units = 0u;
      const bool draining = slot.drain_requested.load(std::memory_order_acquire);
      if (draining) {
        slot.session_id.store(0u, std::memory_order_relaxed);
        slot.generation.store(0u, std::memory_order_relaxed);
      }
      slot.state.store(draining ? LaneState::kDrained : LaneState::kIdle,
                       std::memory_order_release);
      slot.state.notify_one();
      ready_cursor_ = (index + 1uz) % slots_.size();
      pump();
      continue;
    }
    if (job_result_code(slot.result) == JobResultCode::kComplete)
      slot.consecutive_failures = 0uz;
    if (!slot.events_published) {
      const std::uint64_t started_ns = slot.started_ns.load(std::memory_order_relaxed);
      const std::uint64_t finished_ns = slot.finished_ns.load(std::memory_order_relaxed);
      const std::uint64_t preempted_ns = slot.preempted_ns.load(std::memory_order_relaxed);
      const JobResultCode code = job_result_code(slot.result);
      if (!rejected(code)) {
        const std::uint32_t started_worker = slot.migration_source == kNoWorker
                                                 ? static_cast<std::uint32_t>(index)
                                                 : slot.migration_source;
        emit(slot.request.envelope.sequence,
             JobEventDetails{.event = JobEventKind::kStarted,
                             .descriptor = slot.request.metadata.descriptor,
                             .progress =
                                 JobProgress{.state = JobState::kRunning,
                                             .completed_work_units = 0u,
                                             .remaining_work_units =
                                                 slot.request.metadata.descriptor.total_work_units},
                             .timestamp_ns = started_ns,
                             .worker_index = started_worker,
                             .service_class = slot.request.metadata.service_class});
        if (slot.migration_source != kNoWorker) {
          const JobProgress migration_progress{
              .state = JobState::kRunning,
              .completed_work_units = slot.resume_completed_work_units,
              .remaining_work_units = slot.request.metadata.descriptor.total_work_units -
                                      slot.resume_completed_work_units,
          };
          emit(slot.request.envelope.sequence,
               JobEventDetails{.event = JobEventKind::kMigrationStarted,
                               .descriptor = slot.request.metadata.descriptor,
                               .progress = migration_progress,
                               .timestamp_ns = slot.migration_started_ns,
                               .worker_index = slot.migration_source,
                               .peer_worker_index = static_cast<std::uint32_t>(index),
                               .service_class = slot.request.metadata.service_class});
          if (slot.migration_completed_ns != 0u) {
            emit(slot.request.envelope.sequence,
                 JobEventDetails{.event = JobEventKind::kMigrationCompleted,
                                 .descriptor = slot.request.metadata.descriptor,
                                 .progress = migration_progress,
                                 .timestamp_ns = slot.migration_completed_ns,
                                 .worker_index = static_cast<std::uint32_t>(index),
                                 .peer_worker_index = slot.migration_source,
                                 .service_class = slot.request.metadata.service_class});
          }
        }
      }
      const JobEventKind terminal_event =
          code == JobResultCode::kComplete    ? JobEventKind::kCompleted
          : code == JobResultCode::kCancelled ? JobEventKind::kCancelled
          : rejected(code)                    ? JobEventKind::kRejected
                                              : JobEventKind::kFailed;
      emit(slot.result.envelope.sequence,
           JobEventDetails{
               .event = terminal_event,
               .descriptor = slot.result.metadata.descriptor,
               .progress = slot.result.metadata.progress,
               .timestamp_ns = finished_ns,
               .timing = JobEventTiming{.queue_ns =
                                            elapsed(slot.request.metadata.timestamp_ns, started_ns),
                                        .execution_ns = elapsed(started_ns, finished_ns),
                                        .end_to_end_ns = elapsed(slot.request.metadata.timestamp_ns,
                                                                 finished_ns),
                                        .cancellation_ns = elapsed(preempted_ns, finished_ns)},
               .worker_index = static_cast<std::uint32_t>(index),
               .result_code = code,
               .service_class = slot.request.metadata.service_class});
      slot.events_published = true;
    }
    ready_index_ = index;
    return &slot.result;
  }
  return nullptr;
}

JobWorkerTiming JobWorkerPool::ready_timing() const noexcept
{
  if (!ready_index_)
    return {};
  const Slot& slot = *slots_[*ready_index_];
  if (slot.state.load(std::memory_order_acquire) != LaneState::kReady)
    return {};
  return JobWorkerTiming{
      .dispatched_ns = slot.dispatched_ns.load(std::memory_order_relaxed),
      .started_ns = slot.started_ns.load(std::memory_order_relaxed),
      .finished_ns = slot.finished_ns.load(std::memory_order_relaxed),
  };
}

void JobWorkerPool::release_ready_result() noexcept
{
  if (!ready_index_)
    return;
  Slot& slot = *slots_[*ready_index_];
  ready_cursor_ = (*ready_index_ + 1uz) % slots_.size();
  ready_index_.reset();
  slot.remaining_work_units.store(0u, std::memory_order_relaxed);
  slot.migration_source = kNoWorker;
  slot.resume_completed_work_units = 0u;
  const bool draining = slot.drain_requested.load(std::memory_order_acquire);
  if (draining) {
    slot.session_id.store(0u, std::memory_order_relaxed);
    slot.generation.store(0u, std::memory_order_relaxed);
  }
  slot.state.store(draining ? LaneState::kDrained : LaneState::kIdle, std::memory_order_release);
  slot.state.notify_one();
  pump();
}

void JobWorkerPool::stop() noexcept
{
  for (const auto& slot : slots_) {
    slot->stop_requested.store(true, std::memory_order_release);
    update_max(slot->cancel_generation, std::numeric_limits<std::uint64_t>::max());
    slot->state.store(LaneState::kRequested, std::memory_order_release);
    slot->state.notify_one();
  }
  for (const auto& slot : slots_)
    if (slot->thread.joinable())
      slot->thread.join();
  slots_.clear();
  queue_order_.clear();
  ready_index_.reset();
}

} // namespace fe::relay
