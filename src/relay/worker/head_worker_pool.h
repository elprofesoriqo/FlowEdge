#pragma once

#include "relay/worker/head_worker.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace fe::relay {

// A fixed-size set of independent Core engines. Each slot owns its request and
// result buffers and executes on one dedicated outer worker thread. Core's own
// thread count remains independently configurable per engine.
class HeadWorkerPool
{
public:
  [[nodiscard]] static std::expected<HeadWorkerPool, std::string> open(
      std::string_view model_path, std::size_t worker_count,
      std::optional<unsigned> threads_per_worker = std::nullopt) noexcept;

  ~HeadWorkerPool();
  HeadWorkerPool(const HeadWorkerPool&) = delete;
  HeadWorkerPool& operator=(const HeadWorkerPool&) = delete;
  HeadWorkerPool(HeadWorkerPool&& other) noexcept;
  HeadWorkerPool& operator=(HeadWorkerPool&&) = delete;

  [[nodiscard]] const fe_model_metadata& model_metadata() const noexcept { return metadata_; }
  [[nodiscard]] std::size_t worker_count() const noexcept { return slots_.size(); }
  [[nodiscard]] std::size_t busy_count() const noexcept;
  [[nodiscard]] std::uint64_t failure_count() const noexcept { return failure_count_; }
  [[nodiscard]] bool has_idle() const noexcept;
  [[nodiscard]] std::uint64_t active_remaining_nfe_at_or_after(
      std::uint64_t generation) const noexcept;
  [[nodiscard]] const std::string& last_error() const noexcept { return last_error_; }

  [[nodiscard]] bool try_dispatch(const ConditionMessage& request) noexcept;
  void cancel_before(std::uint64_t generation) noexcept;

  // The returned view remains valid until release_ready_action().
  [[nodiscard]] const ActionMessage* ready_action() noexcept;
  void release_ready_action() noexcept;

private:
  struct Slot;

  HeadWorkerPool() = default;
  static void run_slot(Slot& slot) noexcept;
  void stop() noexcept;

  std::vector<std::unique_ptr<Slot>> slots_{};
  fe_model_metadata metadata_{};
  std::optional<std::size_t> ready_index_{};
  std::size_t dispatch_cursor_{};
  std::size_t ready_cursor_{};
  std::uint64_t failure_count_{};
  std::string last_error_{};
};

} // namespace fe::relay
