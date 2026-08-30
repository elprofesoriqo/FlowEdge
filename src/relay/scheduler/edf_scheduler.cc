#include "relay/scheduler/edf_scheduler.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace fe::relay {
namespace {

[[nodiscard]] constexpr std::uint64_t deadline_key(const ConditionMessage& message) noexcept
{
  return message.metadata.deadline_ns == 0u ? std::numeric_limits<std::uint64_t>::max()
                                            : message.metadata.deadline_ns;
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

} // namespace

EdfScheduler::EdfScheduler(std::size_t capacity, AdmissionPolicy policy)
    : capacity_{std::max(capacity, 1uz)}, admission_{policy}, storage_(capacity_)
{
  heap_.reserve(capacity_);
  free_slots_.reserve(capacity_);
  admission_jobs_.reserve(capacity_ + 1uz);
  for (std::size_t slot{capacity_}; slot > 0uz; --slot)
    free_slots_.push_back(slot - 1uz);
}

bool EdfScheduler::admissible(const ConditionMessage& message, AdmissionContext context) noexcept
{
  if (admission_.nanoseconds_per_nfe == 0u || context.now_ns == 0u ||
      message.metadata.deadline_ns == 0u)
    return true;
  if (message.metadata.deadline_ns <= context.now_ns)
    return false;

  std::optional<std::size_t> displaced{};
  if (heap_.size() >= capacity_) {
    const auto worst = std::ranges::max_element(heap_, [this](std::size_t left, std::size_t right) {
      return earlier(storage_[left], storage_[right]);
    });
    if (worst != heap_.end() && earlier(message, storage_[*worst]))
      displaced = *worst;
  }

  const std::size_t worker_count =
      std::clamp(context.worker_count, 1uz, AdmissionContext::kMaxWorkers);
  std::array<std::uint64_t, AdmissionContext::kMaxWorkers> available_at{};
  std::ranges::fill(available_at, context.now_ns);
  for (std::size_t lane{}; lane < worker_count; ++lane) {
    const AdmissionContext::ActiveLane& active = context.active[lane];
    if (active.remaining_nfe == 0u || active.generation < message.metadata.generation)
      continue;
    available_at[lane] =
        saturating_add(context.now_ns,
                       saturating_multiply(active.remaining_nfe, admission_.nanoseconds_per_nfe));
  }

  admission_jobs_.clear();
  admission_jobs_.push_back(&message);
  for (const std::size_t slot : heap_) {
    if (displaced && slot == *displaced)
      continue;
    const ConditionMessage& queued = storage_[slot];
    if (queued.metadata.generation >= message.metadata.generation &&
        queued.metadata.deadline_ns != 0u)
      admission_jobs_.push_back(&queued);
  }
  std::ranges::sort(admission_jobs_,
                    [](const ConditionMessage* left, const ConditionMessage* right) {
                      return EdfScheduler::earlier(*left, *right);
                    });
  for (const ConditionMessage* const job : admission_jobs_) {
    auto lane = std::ranges::min_element(available_at.begin(), available_at.begin() + worker_count);
    const std::uint64_t duration =
        saturating_multiply(job->metadata.remaining_nfe, admission_.nanoseconds_per_nfe);
    if (*lane > job->metadata.deadline_ns || duration > job->metadata.deadline_ns - *lane)
      return false;
    const std::uint64_t finish = *lane + duration;
    if (admission_.reserve_ns > job->metadata.deadline_ns - finish)
      return false;
    *lane = finish;
  }
  return true;
}

bool EdfScheduler::later(const ConditionMessage& left, const ConditionMessage& right) noexcept
{
  const std::uint64_t left_deadline = deadline_key(left);
  const std::uint64_t right_deadline = deadline_key(right);
  if (left_deadline != right_deadline)
    return left_deadline > right_deadline;
  if (left.metadata.generation != right.metadata.generation)
    return left.metadata.generation < right.metadata.generation;
  return left.envelope.sequence > right.envelope.sequence;
}

bool EdfScheduler::earlier(const ConditionMessage& left, const ConditionMessage& right) noexcept
{
  return later(right, left);
}

void EdfScheduler::rebuild() noexcept
{
  std::ranges::make_heap(heap_, [this](std::size_t left, std::size_t right) {
    return later(storage_[left], storage_[right]);
  });
}

void EdfScheduler::copy_into(std::size_t slot, const ConditionMessage& message) noexcept
{
  std::memcpy(&storage_[slot], &message, wire_size(message));
}

void EdfScheduler::release_borrowed() noexcept
{
  if (borrowed_) {
    free_slots_.push_back(*borrowed_);
    borrowed_.reset();
  }
}

SubmitResult EdfScheduler::submit(const ConditionMessage& message, AdmissionContext context,
                                  ActionMessage* displaced) noexcept
{
  release_borrowed();
  if (validate(message) != ProtocolResult::kSuccess) {
    return SubmitResult::kInvalid;
  }
  const std::uint64_t generation = message.metadata.generation;
  if (generation < newest_generation_) {
    ++stats_.stale;
    return SubmitResult::kStale;
  }
  if (!admissible(message, context)) {
    ++stats_.unreachable;
    return SubmitResult::kDeadlineUnreachable;
  }
  if (generation > newest_generation_) {
    newest_generation_ = generation;
    const std::size_t before = heap_.size();
    std::erase_if(heap_, [&](std::size_t slot) {
      if (storage_[slot].metadata.generation >= newest_generation_)
        return false;
      free_slots_.push_back(slot);
      return true;
    });
    stats_.stale += before - heap_.size();
    rebuild();
  }

  if (heap_.size() < capacity_) {
    const std::size_t slot = free_slots_.back();
    free_slots_.pop_back();
    copy_into(slot, message);
    heap_.push_back(slot);
    std::ranges::push_heap(heap_, [this](std::size_t left, std::size_t right) {
      return later(storage_[left], storage_[right]);
    });
    ++stats_.accepted;
    return SubmitResult::kAccepted;
  }

  const auto worst = std::ranges::max_element(heap_, [this](std::size_t left, std::size_t right) {
    return earlier(storage_[left], storage_[right]);
  });
  if (worst != heap_.end() && earlier(message, storage_[*worst])) {
    if (displaced != nullptr)
      make_rejected_action(*displaced, storage_[*worst], context.now_ns,
                           RelayActionCode::kRejectedCapacity);
    copy_into(*worst, message);
    rebuild();
    ++stats_.accepted;
    ++stats_.evicted;
    return SubmitResult::kAcceptedAndEvicted;
  }
  ++stats_.full;
  return SubmitResult::kFull;
}

const ConditionMessage* EdfScheduler::pop(std::uint64_t now_ns, ActionMessage* rejection) noexcept
{
  release_borrowed();
  while (!heap_.empty()) {
    std::ranges::pop_heap(heap_, [this](std::size_t left, std::size_t right) {
      return later(storage_[left], storage_[right]);
    });
    const std::size_t slot = heap_.back();
    heap_.pop_back();
    const ConditionMessage& message = storage_[slot];
    if (message.metadata.generation < newest_generation_) {
      ++stats_.stale;
      free_slots_.push_back(slot);
      continue;
    }
    if (message.metadata.deadline_ns != 0u && message.metadata.deadline_ns <= now_ns) {
      ++stats_.expired;
      if (rejection != nullptr)
        make_rejected_action(*rejection, message, now_ns, RelayActionCode::kExpired);
      free_slots_.push_back(slot);
      if (rejection != nullptr)
        return nullptr;
      continue;
    }
    borrowed_ = slot;
    return &message;
  }
  return nullptr;
}

} // namespace fe::relay
