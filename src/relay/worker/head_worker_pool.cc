#include "relay/worker/head_worker_pool.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <ranges>
#include <thread>
#include <utility>

namespace fe::relay {
namespace {

enum class SlotState : std::uint8_t
{
  kIdle,
  kRequested,
  kRunning,
  kReady,
};

[[nodiscard]] bool same_model(const fe_model_metadata& left,
                              const fe_model_metadata& right) noexcept
{
  return left.struct_size == right.struct_size && left.protocol_version == right.protocol_version &&
         left.architecture == right.architecture && left.precision == right.precision &&
         left.d_model == right.d_model && left.n_layers == right.n_layers &&
         left.d_inner == right.d_inner && left.d_state == right.d_state &&
         left.d_conv == right.d_conv && left.action_dim == right.action_dim &&
         left.condition_dim == right.condition_dim &&
         left.decode_snapshot_bytes == right.decode_snapshot_bytes &&
         std::ranges::equal(left.model_digest.bytes, right.model_digest.bytes);
}

void update_max(std::atomic<std::uint64_t>& value, std::uint64_t candidate) noexcept
{
  std::uint64_t current = value.load(std::memory_order_relaxed);
  while (current < candidate &&
         !value.compare_exchange_weak(current, candidate, std::memory_order_release,
                                      std::memory_order_relaxed)) {
  }
}

} // namespace

struct HeadWorkerPool::Slot
{
  explicit Slot(HeadWorker source) noexcept : worker{std::move(source)} {}

  HeadWorker worker;
  ConditionMessage request{};
  ActionMessage result{};
  std::atomic<SlotState> state{SlotState::kIdle};
  std::atomic_bool stop_requested{false};
  std::atomic<std::uint64_t> generation{};
  std::atomic<std::uint64_t> remaining_nfe{};
  std::atomic<std::uint64_t> cancel_generation{};
  bool execution_failed{};
  std::thread thread{};
};

void HeadWorkerPool::run_slot(Slot& slot) noexcept
{
  while (!slot.stop_requested.load(std::memory_order_acquire)) {
    SlotState state = slot.state.load(std::memory_order_acquire);
    if (state != SlotState::kRequested) {
      slot.state.wait(state, std::memory_order_relaxed);
      continue;
    }

    slot.result = {};
    slot.execution_failed = false;
    if (!slot.worker.begin(slot.request)) {
      slot.execution_failed = true;
      slot.remaining_nfe.store(0u, std::memory_order_release);
    } else {
      slot.state.store(SlotState::kRunning, std::memory_order_release);
      slot.worker.cancel_before(slot.cancel_generation.load(std::memory_order_acquire));
      WorkerStep step{WorkerStep::kInProgress};
      while (step == WorkerStep::kInProgress &&
             !slot.stop_requested.load(std::memory_order_acquire)) {
        step = slot.worker.advance(slot.result);
        slot.remaining_nfe.store(slot.worker.remaining_nfe(), std::memory_order_release);
      }
      slot.execution_failed = step == WorkerStep::kError;
      if (slot.stop_requested.load(std::memory_order_acquire)) {
        slot.worker.cancel_before(std::numeric_limits<std::uint64_t>::max());
        if (slot.worker.busy())
          static_cast<void>(slot.worker.advance(slot.result));
      }
    }

    if (slot.stop_requested.load(std::memory_order_acquire))
      break;
    slot.state.store(SlotState::kReady, std::memory_order_release);
    slot.state.notify_one();
  }
}

std::expected<HeadWorkerPool, std::string> HeadWorkerPool::open(
    std::string_view model_path, std::size_t worker_count,
    std::optional<unsigned> threads_per_worker) noexcept
{
  if (worker_count == 0uz || worker_count > 8uz)
    return std::unexpected("Relay worker count must be in the range 1..8");
  try {
    HeadWorkerPool pool{};
    pool.slots_.reserve(worker_count);
    for (std::size_t i{0uz}; i < worker_count; ++i) {
      auto opened = HeadWorker::open(model_path, threads_per_worker);
      if (!opened)
        return std::unexpected(opened.error());
      if (i == 0uz)
        pool.metadata_ = opened->model_metadata();
      else if (!same_model(pool.metadata_, opened->model_metadata()))
        return std::unexpected("Relay worker pool loaded inconsistent model identities");
      pool.slots_.push_back(std::make_unique<Slot>(std::move(*opened)));
    }
    for (const auto& slot : pool.slots_)
      slot->thread = std::thread{HeadWorkerPool::run_slot, std::ref(*slot)};
    return pool;
  } catch (const std::exception& error) {
    return std::unexpected("Failed to create Relay worker pool: " + std::string{error.what()});
  } catch (...) {
    return std::unexpected("Failed to create Relay worker pool");
  }
}

HeadWorkerPool::~HeadWorkerPool()
{
  stop();
}

HeadWorkerPool::HeadWorkerPool(HeadWorkerPool&& other) noexcept
    : slots_{std::move(other.slots_)}, metadata_{other.metadata_}, ready_index_{other.ready_index_},
      dispatch_cursor_{other.dispatch_cursor_}, ready_cursor_{other.ready_cursor_},
      failure_count_{other.failure_count_}, last_error_{std::move(other.last_error_)}
{
  other.ready_index_.reset();
}

std::size_t HeadWorkerPool::busy_count() const noexcept
{
  return static_cast<std::size_t>(std::ranges::count_if(slots_, [](const auto& slot) {
    const SlotState state = slot->state.load(std::memory_order_acquire);
    return state == SlotState::kRequested || state == SlotState::kRunning;
  }));
}

bool HeadWorkerPool::has_idle() const noexcept
{
  return std::ranges::any_of(slots_, [](const auto& slot) {
    return slot->state.load(std::memory_order_acquire) == SlotState::kIdle;
  });
}

std::uint64_t HeadWorkerPool::active_remaining_nfe_at_or_after(
    std::uint64_t generation) const noexcept
{
  std::uint64_t total{};
  for (const auto& slot : slots_) {
    const SlotState state = slot->state.load(std::memory_order_acquire);
    if ((state != SlotState::kRequested && state != SlotState::kRunning) ||
        slot->generation.load(std::memory_order_acquire) < generation)
      continue;
    const std::uint64_t remaining = slot->remaining_nfe.load(std::memory_order_acquire);
    total = remaining > std::numeric_limits<std::uint64_t>::max() - total
                ? std::numeric_limits<std::uint64_t>::max()
                : total + remaining;
  }
  return total;
}

bool HeadWorkerPool::try_dispatch(const ConditionMessage& request) noexcept
{
  last_error_.clear();
  if (validate(request) != ProtocolResult::kSuccess) {
    last_error_ = "Relay worker pool request is invalid";
    return false;
  }
  for (std::size_t checked{0uz}; checked < slots_.size(); ++checked) {
    const std::size_t index = (dispatch_cursor_ + checked) % slots_.size();
    Slot& slot = *slots_[index];
    if (slot.state.load(std::memory_order_acquire) != SlotState::kIdle)
      continue;
    std::memcpy(&slot.request, &request, wire_size(request));
    slot.generation.store(request.metadata.generation, std::memory_order_relaxed);
    slot.remaining_nfe.store(request.metadata.remaining_nfe, std::memory_order_relaxed);
    slot.state.store(SlotState::kRequested, std::memory_order_release);
    slot.state.notify_one();
    dispatch_cursor_ = (index + 1uz) % slots_.size();
    return true;
  }
  return false;
}

void HeadWorkerPool::cancel_before(std::uint64_t generation) noexcept
{
  for (std::size_t index{0uz}; index < slots_.size(); ++index) {
    Slot& slot = *slots_[index];
    update_max(slot.cancel_generation, generation);
    const SlotState state = slot.state.load(std::memory_order_acquire);
    if (state == SlotState::kRunning)
      slot.worker.cancel_before(generation);
    if (state == SlotState::kReady &&
        slot.generation.load(std::memory_order_acquire) < generation) {
      SlotState expected{SlotState::kReady};
      if (slot.state.compare_exchange_strong(expected, SlotState::kIdle,
                                             std::memory_order_acq_rel)) {
        if (ready_index_ == index)
          ready_index_.reset();
        slot.state.notify_one();
      }
    }
  }
}

const ActionMessage* HeadWorkerPool::ready_action() noexcept
{
  if (ready_index_) {
    if (slots_[*ready_index_]->state.load(std::memory_order_acquire) == SlotState::kReady)
      return &slots_[*ready_index_]->result;
    ready_index_.reset();
  }
  for (std::size_t checked{0uz}; checked < slots_.size(); ++checked) {
    const std::size_t index = (ready_cursor_ + checked) % slots_.size();
    Slot& slot = *slots_[index];
    if (slot.state.load(std::memory_order_acquire) != SlotState::kReady)
      continue;
    if (slot.execution_failed) {
      ++failure_count_;
      last_error_ =
          slot.worker.last_error().empty() ? "Relay pool worker failed" : slot.worker.last_error();
      slot.execution_failed = false;
    }
    if (validate(slot.result) != ProtocolResult::kSuccess) {
      slot.state.store(SlotState::kIdle, std::memory_order_release);
      slot.state.notify_one();
      ready_cursor_ = (index + 1uz) % slots_.size();
      continue;
    }
    ready_index_ = index;
    return &slot.result;
  }
  return nullptr;
}

void HeadWorkerPool::release_ready_action() noexcept
{
  if (!ready_index_)
    return;
  Slot& slot = *slots_[*ready_index_];
  ready_cursor_ = (*ready_index_ + 1uz) % slots_.size();
  ready_index_.reset();
  slot.remaining_nfe.store(0u, std::memory_order_relaxed);
  slot.state.store(SlotState::kIdle, std::memory_order_release);
  slot.state.notify_one();
}

void HeadWorkerPool::stop() noexcept
{
  for (const auto& slot : slots_) {
    slot->stop_requested.store(true, std::memory_order_release);
    update_max(slot->cancel_generation, std::numeric_limits<std::uint64_t>::max());
    const SlotState state = slot->state.load(std::memory_order_acquire);
    if (state == SlotState::kRunning)
      slot->worker.cancel_before(std::numeric_limits<std::uint64_t>::max());
    else
      slot->state.store(SlotState::kRequested, std::memory_order_release);
    slot->state.notify_one();
  }
  for (const auto& slot : slots_)
    if (slot->thread.joinable())
      slot->thread.join();
  slots_.clear();
  ready_index_.reset();
}

} // namespace fe::relay
