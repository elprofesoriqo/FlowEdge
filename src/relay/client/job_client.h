#pragma once

#include "relay/client/client_result.h"
#include "relay/protocol/job_messages.h"
#include "relay/shared_memory/shared_memory_ring.h"

#include <cstdint>
#include <expected>
#include <string>
#include <utility>

namespace fe::relay {

// Typed, allocation-free SPSC endpoint for generic cooperative jobs. One
// client owns one request producer and one result consumer.
class JobClient
{
public:
  [[nodiscard]] static std::expected<JobClient, std::string> create(
      std::string request_name, std::string result_name, std::uint32_t capacity = 32u) noexcept;
  [[nodiscard]] static std::expected<JobClient, std::string> connect(
      std::string request_name, std::string result_name) noexcept;
  [[nodiscard]] static std::expected<ClientResult, std::string> send_shutdown(
      std::string request_name, std::uint64_t sequence, std::uint64_t session_id,
      std::uint64_t reason = 0u) noexcept;

  JobClient(const JobClient&) = delete;
  JobClient& operator=(const JobClient&) = delete;
  JobClient(JobClient&&) noexcept = default;
  JobClient& operator=(JobClient&&) noexcept = default;

  [[nodiscard]] ClientResult try_submit(const JobRequestMessage& request) noexcept;
  [[nodiscard]] ClientResult try_receive(JobResultMessage& result) noexcept;
  [[nodiscard]] ClientResult try_shutdown(std::uint64_t sequence, std::uint64_t session_id,
                                          std::uint64_t reason = 0u) noexcept;

  [[nodiscard]] std::uint64_t pending_requests() const noexcept { return requests_.size(); }
  [[nodiscard]] std::uint64_t pending_results() const noexcept { return results_.size(); }

private:
  JobClient(SharedMemoryRing requests, SharedMemoryRing results) noexcept
      : requests_{std::move(requests)}, results_{std::move(results)}
  {
  }

  [[nodiscard]] static ClientResult map_ring_result(RingResult result) noexcept;

  SharedMemoryRing requests_{};
  SharedMemoryRing results_{};
};

} // namespace fe::relay
