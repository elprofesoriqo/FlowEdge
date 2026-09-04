#pragma once

#include "relay/client/client_result.h"
#include "relay/protocol/job_control.h"
#include "relay/shared_memory/shared_memory_ring.h"

#include <cstdint>
#include <expected>
#include <string>
#include <utility>

namespace fe::relay {

// Single-producer/single-consumer administration endpoint. Keep one request
// outstanding so responses retain their request order.
class JobControlClient
{
public:
  [[nodiscard]] static std::expected<JobControlClient, std::string> create(
      std::string request_name, std::string response_name, std::uint32_t capacity = 8u) noexcept;
  [[nodiscard]] static std::expected<JobControlClient, std::string> connect(
      std::string request_name, std::string response_name) noexcept;

  JobControlClient(const JobControlClient&) = delete;
  JobControlClient& operator=(const JobControlClient&) = delete;
  ~JobControlClient() = default;
  JobControlClient(JobControlClient&&) noexcept = default;
  JobControlClient& operator=(JobControlClient&&) noexcept = default;

  [[nodiscard]] ClientResult try_submit(const JobControlRequest& request) noexcept;
  [[nodiscard]] ClientResult try_receive(JobControlResponse& response) noexcept;

private:
  JobControlClient(SharedMemoryRing requests, SharedMemoryRing responses) noexcept
      : requests_{std::move(requests)}, responses_{std::move(responses)}
  {
  }

  [[nodiscard]] static ClientResult map_ring_result(RingResult result) noexcept;

  SharedMemoryRing requests_{};
  SharedMemoryRing responses_{};
};

} // namespace fe::relay
