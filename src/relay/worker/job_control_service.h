#pragma once

#include "relay/protocol/job_control.h"
#include "relay/shared_memory/shared_memory_ring.h"
#include "relay/worker/job_service.h"
#include "relay/worker/job_worker_pool.h"

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <utility>

namespace fe::relay {

enum class JobControlServiceResult : std::uint8_t
{
  kIdle,
  kProgress,
  kBackpressured,
  kStopped,
  kCorruptInput,
  kTransportError,
};

class JobControlService
{
public:
  [[nodiscard]] static std::expected<JobControlService, std::string> create(
      std::string request_name, std::string response_name, JobWorkerPool& pool,
      const JobTransportStats& transport_stats, std::uint32_t capacity = 8u) noexcept;
  [[nodiscard]] static std::expected<JobControlService, std::string> connect(
      std::string request_name, std::string response_name, JobWorkerPool& pool,
      const JobTransportStats& transport_stats) noexcept;

  JobControlService(const JobControlService&) = delete;
  JobControlService& operator=(const JobControlService&) = delete;
  ~JobControlService() = default;
  JobControlService(JobControlService&&) noexcept = default;
  JobControlService& operator=(JobControlService&&) noexcept = default;

  [[nodiscard]] JobControlServiceResult poll() noexcept;
  [[nodiscard]] bool stopped() const noexcept { return stopped_; }

private:
  JobControlService(SharedMemoryRing requests, SharedMemoryRing responses, JobWorkerPool& pool,
                    const JobTransportStats& transport_stats) noexcept
      : requests_{std::move(requests)}, responses_{std::move(responses)}, pool_{&pool},
        transport_stats_{&transport_stats}
  {
  }

  void build_response(const JobControlRequest& request) noexcept;
  void capture_status() noexcept;
  [[nodiscard]] JobControlServiceResult publish_pending() noexcept;

  SharedMemoryRing requests_{};
  SharedMemoryRing responses_{};
  JobWorkerPool* pool_{};
  const JobTransportStats* transport_stats_{};
  JobControlRequest request_{};
  JobControlResponse response_{};
  bool response_pending_{};
  bool stop_after_publish_{};
  bool stopped_{};
};

[[nodiscard]] std::string_view to_string(JobControlServiceResult result) noexcept;

} // namespace fe::relay
