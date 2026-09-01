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
                      timestamp_ns, code);
  if (written != ProtocolResult::kSuccess)
    result = {};
}

} // namespace

enum class JobWorkerPool::LaneState : std::uint8_t
{
  kIdle,
  kRequested,
  kRunning,
  kReady,
};

struct JobWorkerPool::Slot
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
  std::atomic<LaneState> state{LaneState::kIdle};
  std::atomic_bool stop_requested{false};
  std::atomic<std::uint64_t> session_id{};
  std::atomic<std::uint64_t> generation{};
  std::atomic<std::uint64_t> remaining_work_units{};
  std::atomic<std::uint64_t> cancel_generation{};
  std::atomic<std::uint64_t> dispatched_ns{};
  std::atomic<std::uint64_t> started_ns{};
  std::atomic<std::uint64_t> finished_ns{};
  std::atomic<std::uint32_t> numa_node{};
  std::atomic<std::uint32_t> logical_cpu{};
  std::atomic_bool bound{false};
  bool execution_failed{};
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

    slot.result = {};
    slot.execution_failed = false;
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
      auto progress = routed->start();
      if (!progress) {
        slot.execution_failed = true;
      } else {
        const std::uint64_t cancel_generation =
            slot.cancel_generation.load(std::memory_order_acquire);
        if (cancel_generation > routed->descriptor().generation)
          progress = routed->cancel_before(cancel_generation);
      }

      while (progress && progress->state == JobState::kRunning &&
             !slot.stop_requested.load(std::memory_order_acquire)) {
        progress = routed->advance(slot.work_quantum,
                                   slot.cancel_generation.load(std::memory_order_acquire));
        if (progress) {
          slot.remaining_work_units.store(progress->remaining_work_units,
                                          std::memory_order_release);
        } else {
          slot.execution_failed = true;
        }
      }

      if (slot.stop_requested.load(std::memory_order_acquire)) {
        static_cast<void>(routed->cancel_before(std::numeric_limits<std::uint64_t>::max()));
        break;
      }
      if (routed->write_result(slot.result, monotonic_ns()) != ProtocolResult::kSuccess) {
        make_failed_result(slot.result, slot.request, monotonic_ns(), JobResultCode::kFailed);
        slot.execution_failed = true;
      }
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
    WorkerPlacement placement) noexcept
{
  if (lane_registries.empty() || lane_registries.size() > 8uz)
    return std::unexpected("Generic Relay worker count must be in the range 1..8");
  if (queue_capacity == 0uz || max_sessions == 0uz || work_quantum == 0uz)
    return std::unexpected("Generic Relay queue, session, and work limits must be non-zero");
  for (const JobAdapterRegistry& registry : lane_registries)
    if (!registry.frozen() || registry.size() == 0uz)
      return std::unexpected("Every generic Relay worker lane requires a frozen adapter registry");

  try {
    JobWorkerPool pool{};
    pool.cost_policy_ = cost_policy;
    pool.work_quantum_ = work_quantum;
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
      cost_policy_{other.cost_policy_}, work_quantum_{other.work_quantum_},
      ready_index_{other.ready_index_}, dispatch_cursor_{other.dispatch_cursor_},
      ready_cursor_{other.ready_cursor_},
      failure_count_{other.failure_count_.load(std::memory_order_relaxed)}
{
  other.ready_index_.reset();
}

std::size_t JobWorkerPool::busy_count() const noexcept
{
  return static_cast<std::size_t>(std::ranges::count_if(slots_, [](const auto& slot) {
    const LaneState state = slot->state.load(std::memory_order_acquire);
    return state == LaneState::kRequested || state == LaneState::kRunning;
  }));
}

bool JobWorkerPool::has_idle() const noexcept
{
  return std::ranges::any_of(slots_, [](const auto& slot) {
    return slot->state.load(std::memory_order_acquire) == LaneState::kIdle;
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
    *destination = {};
    static_cast<void>(make_rejected_job_result(*destination, request, now_ns, code));
  }
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
  if (stale(request)) {
    publish_rejection(rejection, request, now_ns, JobResultCode::kRejectedStale);
    return JobSubmitResult::kStale;
  }
  if (!has_session_capacity(request.metadata.descriptor.session_id) ||
      queue_order_.size() == queue_storage_.size()) {
    publish_rejection(rejection, request, now_ns, JobResultCode::kRejectedCapacity);
    return JobSubmitResult::kFull;
  }
  if (!admissible(request, now_ns)) {
    publish_rejection(rejection, request, now_ns, JobResultCode::kRejectedDeadline);
    return JobSubmitResult::kDeadlineUnreachable;
  }

  const std::size_t queue_slot = free_queue_slots_.back();
  free_queue_slots_.pop_back();
  std::memcpy(&queue_storage_[queue_slot], &request, wire_size(request));
  queue_order_.push_back(queue_slot);

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
    if (state == LaneState::kRequested || state == LaneState::kRunning) {
      update_max(slot.cancel_generation, watermark->generation);
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
  std::optional<std::size_t> selected_position{};
  for (std::size_t position{}; position < queue_order_.size(); ++position) {
    const JobRequestMessage& request = queue_storage_[queue_order_[position]];
    const bool rejected =
        stale(request) || (request.metadata.descriptor.deadline_ns != 0u &&
                           request.metadata.descriptor.deadline_ns <= monotonic_ns());
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

  const std::uint64_t dispatched_ns = monotonic_ns();
  slot.session_id.store(request.metadata.descriptor.session_id, std::memory_order_relaxed);
  slot.generation.store(request.metadata.descriptor.generation, std::memory_order_relaxed);
  slot.remaining_work_units.store(request.metadata.descriptor.total_work_units,
                                  std::memory_order_relaxed);
  slot.cancel_generation.store(0u, std::memory_order_relaxed);
  slot.dispatched_ns.store(dispatched_ns, std::memory_order_relaxed);
  slot.started_ns.store(0u, std::memory_order_relaxed);
  slot.finished_ns.store(0u, std::memory_order_relaxed);

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
  if (queue_order_.empty())
    return;
  for (std::size_t checked{}; checked < slots_.size() && !queue_order_.empty(); ++checked) {
    const std::size_t index = (dispatch_cursor_ + checked) % slots_.size();
    Slot& slot = *slots_[index];
    if (slot.state.load(std::memory_order_acquire) != LaneState::kIdle)
      continue;
    if (dispatch_one(slot))
      dispatch_cursor_ = (index + 1uz) % slots_.size();
  }
}

const JobResultMessage* JobWorkerPool::ready_result() noexcept
{
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
    if (slot.execution_failed) {
      failure_count_.fetch_add(1u, std::memory_order_relaxed);
      slot.execution_failed = false;
    }
    if (validate(slot.result) != ProtocolResult::kSuccess) {
      failure_count_.fetch_add(1u, std::memory_order_relaxed);
      slot.state.store(LaneState::kIdle, std::memory_order_release);
      slot.state.notify_one();
      ready_cursor_ = (index + 1uz) % slots_.size();
      pump();
      continue;
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
  slot.state.store(LaneState::kIdle, std::memory_order_release);
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
