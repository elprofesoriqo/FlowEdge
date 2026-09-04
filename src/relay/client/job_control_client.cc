#include "relay/client/job_control_client.h"

#include <cstddef>
#include <span>
#include <utility>

namespace fe::relay {
namespace {

[[nodiscard]] bool usable_ring(const SharedMemoryRing& ring, std::size_t slot_bytes) noexcept
{
  const RingConfig config = ring.config();
  return config.capacity >= 2u && config.slot_bytes >= slot_bytes;
}

} // namespace

std::expected<JobControlClient, std::string> JobControlClient::create(
    std::string request_name, std::string response_name, std::uint32_t capacity) noexcept
{
  auto requests = SharedMemoryRing::create(std::move(request_name),
                                           RingConfig{.capacity = capacity,
                                                      .slot_bytes = sizeof(JobControlRequest)});
  if (!requests)
    return std::unexpected(requests.error());
  auto responses = SharedMemoryRing::create(std::move(response_name),
                                            RingConfig{.capacity = capacity,
                                                       .slot_bytes = sizeof(JobControlResponse)});
  if (!responses)
    return std::unexpected(responses.error());
  return JobControlClient{std::move(*requests), std::move(*responses)};
}

std::expected<JobControlClient, std::string> JobControlClient::connect(
    std::string request_name, std::string response_name) noexcept
{
  auto requests = SharedMemoryRing::open(std::move(request_name));
  if (!requests)
    return std::unexpected(requests.error());
  auto responses = SharedMemoryRing::open(std::move(response_name));
  if (!responses)
    return std::unexpected(responses.error());
  if (!usable_ring(*requests, sizeof(JobControlRequest)) ||
      !usable_ring(*responses, sizeof(JobControlResponse)))
    return std::unexpected("Generic job control rings are incompatible with this client build");
  return JobControlClient{std::move(*requests), std::move(*responses)};
}

ClientResult JobControlClient::map_ring_result(RingResult result) noexcept
{
  switch (result) {
  case RingResult::kSuccess:
    return ClientResult::kSuccess;
  case RingResult::kEmpty:
    return ClientResult::kEmpty;
  case RingResult::kFull:
    return ClientResult::kFull;
  case RingResult::kTooLarge:
  case RingResult::kDestinationTooSmall:
    return ClientResult::kIncompatible;
  case RingResult::kCorrupt:
    return ClientResult::kCorrupt;
  }
  return ClientResult::kCorrupt;
}

ClientResult JobControlClient::try_submit(const JobControlRequest& request) noexcept
{
  if (validate(request) != ProtocolResult::kSuccess)
    return ClientResult::kInvalidRequest;
  return map_ring_result(requests_.try_push(std::as_bytes(std::span{&request, 1uz})));
}

ClientResult JobControlClient::try_receive(JobControlResponse& response) noexcept
{
  response = {};
  std::size_t bytes_read{};
  const RingResult result =
      responses_.try_pop(std::as_writable_bytes(std::span{&response, 1uz}), bytes_read);
  if (result != RingResult::kSuccess)
    return map_ring_result(result);
  if (bytes_read != sizeof(response) || validate(response) != ProtocolResult::kSuccess)
    return ClientResult::kCorrupt;
  return ClientResult::kSuccess;
}

} // namespace fe::relay
