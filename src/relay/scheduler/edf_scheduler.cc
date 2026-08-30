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

} // namespace

EdfScheduler::EdfScheduler(std::size_t capacity)
    : capacity_{std::max(capacity, 1uz)}, storage_(capacity_)
{
  heap_.reserve(capacity_);
  free_slots_.reserve(capacity_);
  for (std::size_t slot{capacity_}; slot > 0uz; --slot)
    free_slots_.push_back(slot - 1uz);
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

SubmitResult EdfScheduler::submit(const ConditionMessage& message) noexcept
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
    copy_into(*worst, message);
    rebuild();
    ++stats_.accepted;
    ++stats_.evicted;
    return SubmitResult::kAcceptedAndEvicted;
  }
  ++stats_.full;
  return SubmitResult::kFull;
}

const ConditionMessage* EdfScheduler::pop(std::uint64_t now_ns) noexcept
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
      free_slots_.push_back(slot);
      continue;
    }
    borrowed_ = slot;
    return &message;
  }
  return nullptr;
}

} // namespace fe::relay
