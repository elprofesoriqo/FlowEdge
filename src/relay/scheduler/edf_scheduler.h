#pragma once

#include "relay/protocol/messages.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace fe::relay {

enum class SubmitResult : std::uint8_t
{
  kAccepted,
  kAcceptedAndEvicted,
  kStale,
  kDeadlineUnreachable,
  kInvalid,
  kFull,
};

struct AdmissionPolicy
{
  std::uint64_t nanoseconds_per_nfe{};
  std::uint64_t reserve_ns{};
};

struct AdmissionContext
{
  std::uint64_t now_ns{};
  std::uint64_t active_generation{};
  std::uint64_t active_remaining_nfe{};
};

struct SchedulerStats
{
  std::uint64_t accepted{};
  std::uint64_t stale{};
  std::uint64_t unreachable{};
  std::uint64_t expired{};
  std::uint64_t evicted{};
  std::uint64_t full{};
};

// A bounded, allocation-stable earliest-deadline-first queue. Freshness is a
// hard gate: observing generation N removes every queued request older than N.
class EdfScheduler
{
public:
  explicit EdfScheduler(std::size_t capacity = 32uz, AdmissionPolicy policy = {});

  [[nodiscard]] SubmitResult submit(const ConditionMessage& message, AdmissionContext context = {},
                                    ActionMessage* displaced = nullptr) noexcept;
  // The returned allocation-free view remains valid until the next submit/pop.
  [[nodiscard]] const ConditionMessage* pop(std::uint64_t now_ns,
                                            ActionMessage* rejection = nullptr) noexcept;

  [[nodiscard]] std::size_t size() const noexcept { return heap_.size(); }
  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::uint64_t newest_generation() const noexcept { return newest_generation_; }
  [[nodiscard]] const SchedulerStats& stats() const noexcept { return stats_; }

private:
  [[nodiscard]] static bool later(const ConditionMessage& left,
                                  const ConditionMessage& right) noexcept;
  [[nodiscard]] static bool earlier(const ConditionMessage& left,
                                    const ConditionMessage& right) noexcept;
  [[nodiscard]] bool admissible(const ConditionMessage& message,
                                AdmissionContext context) const noexcept;
  void copy_into(std::size_t slot, const ConditionMessage& message) noexcept;
  void release_borrowed() noexcept;
  void rebuild() noexcept;

  std::size_t capacity_{};
  std::uint64_t newest_generation_{};
  AdmissionPolicy admission_{};
  std::vector<ConditionMessage> storage_{};
  std::vector<std::size_t> heap_{};
  std::vector<std::size_t> free_slots_{};
  std::optional<std::size_t> borrowed_{};
  SchedulerStats stats_{};
};

} // namespace fe::relay
