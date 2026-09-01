#pragma once

#include "relay/protocol/job_messages.h"
#include "relay/shared_memory/shared_memory_ring.h"
#include "relay/worker/job_worker_pool.h"

#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace fe::relay {

enum class JobServiceResult : std::uint8_t
{
  kIdle,
  kProgress,
  kBackpressured,
  kStopped,
  kCorruptInput,
  kTransportError,
};

struct JobTransportStats
{
  std::uint64_t requests_received{};
  std::uint64_t requests_accepted{};
  std::uint64_t requests_rejected{};
  std::uint64_t results_published{};
  std::uint64_t corrupt_inputs{};
  std::uint64_t publish_blocked{};
};

// Bounded coordinator that connects shared-memory job messages to a caller-
// owned worker pool. poll() is single-threaded and allocation-free. A rejected
// or completed result remains owned until the output ring accepts it.
class JobService
{
public:
  [[nodiscard]] static std::expected<JobService, std::string> create(
      std::string request_name, std::string result_name, JobWorkerPool& pool,
      std::uint32_t capacity = 32u) noexcept;
  [[nodiscard]] static std::expected<JobService, std::string> connect(std::string request_name,
                                                                      std::string result_name,
                                                                      JobWorkerPool& pool) noexcept;

  JobService(const JobService&) = delete;
  JobService& operator=(const JobService&) = delete;
  JobService(JobService&&) noexcept = default;
  JobService& operator=(JobService&&) noexcept = default;

  // A zero timestamp samples steady_clock before admission; tests may inject a
  // nonzero monotonic timestamp.
  [[nodiscard]] JobServiceResult poll(std::uint64_t now_ns = 0u) noexcept;

  [[nodiscard]] bool stopped() const noexcept { return stopped_; }
  [[nodiscard]] const JobTransportStats& stats() const noexcept { return stats_; }
  [[nodiscard]] std::uint64_t pending_requests() const noexcept { return requests_.size(); }
  [[nodiscard]] std::uint64_t pending_results() const noexcept { return results_.size(); }

private:
  JobService(SharedMemoryRing requests, SharedMemoryRing results, JobWorkerPool& pool,
             std::unique_ptr<JobRequestMessage> request,
             std::unique_ptr<JobResultMessage> rejection) noexcept
      : requests_{std::move(requests)}, results_{std::move(results)}, pool_{&pool},
        request_{std::move(request)}, rejection_{std::move(rejection)}
  {
  }

  [[nodiscard]] JobServiceResult publish(const JobResultMessage& result) noexcept;

  SharedMemoryRing requests_{};
  SharedMemoryRing results_{};
  JobWorkerPool* pool_{};
  std::unique_ptr<JobRequestMessage> request_{};
  std::unique_ptr<JobResultMessage> rejection_{};
  JobTransportStats stats_{};
  bool rejection_pending_{};
  bool stopped_{};
};

[[nodiscard]] std::string_view to_string(JobServiceResult result) noexcept;

} // namespace fe::relay
