#include "relay/client/job_client.h"

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

std::expected<JobClient, std::string> JobClient::create(std::string request_name,
                                                        std::string result_name,
                                                        std::uint32_t capacity) noexcept
{
  auto requests = SharedMemoryRing::create(std::move(request_name),
                                           RingConfig{.capacity = capacity,
                                                      .slot_bytes = sizeof(JobRequestMessage)});
  if (!requests)
    return std::unexpected(requests.error());
  auto results = SharedMemoryRing::create(std::move(result_name),
                                          RingConfig{.capacity = capacity,
                                                     .slot_bytes = sizeof(JobResultMessage)});
  if (!results)
    return std::unexpected(results.error());
  return JobClient{std::move(*requests), std::move(*results)};
}

std::expected<JobClient, std::string> JobClient::connect(std::string request_name,
                                                         std::string result_name) noexcept
{
  auto requests = SharedMemoryRing::open(std::move(request_name));
  if (!requests)
    return std::unexpected(requests.error());
  auto results = SharedMemoryRing::open(std::move(result_name));
  if (!results)
    return std::unexpected(results.error());
  if (!usable_ring(*requests, sizeof(JobRequestMessage)) ||
      !usable_ring(*results, sizeof(JobResultMessage)))
    return std::unexpected("Generic job ring slot capacity is incompatible with this client build");
  return JobClient{std::move(*requests), std::move(*results)};
}

std::expected<ClientResult, std::string> JobClient::send_shutdown(std::string request_name,
                                                                  std::uint64_t sequence,
                                                                  std::uint64_t session_id,
                                                                  std::uint64_t reason) noexcept
{
  auto requests = SharedMemoryRing::open(std::move(request_name));
  if (!requests)
    return std::unexpected(requests.error());
  if (!usable_ring(*requests, sizeof(ControlMessage)))
    return std::unexpected("Generic job request ring cannot hold a control message");
  return map_ring_result(requests->try_push(make_shutdown_message(sequence, session_id, reason)));
}

ClientResult JobClient::try_submit(const JobRequestMessage& request) noexcept
{
  if (validate(request) != ProtocolResult::kSuccess)
    return ClientResult::kInvalidRequest;
  return map_ring_result(requests_.try_push(wire_bytes(request)));
}

ClientResult JobClient::try_receive(JobResultMessage& result) noexcept
{
  result.envelope = {};
  result.metadata = {};
  std::size_t bytes_read{};
  const RingResult received =
      results_.try_pop(std::as_writable_bytes(std::span{&result, 1uz}), bytes_read);
  if (received != RingResult::kSuccess)
    return map_ring_result(received);
  if (bytes_read != wire_size(result) || validate(result) != ProtocolResult::kSuccess)
    return ClientResult::kCorrupt;
  return ClientResult::kSuccess;
}

ClientResult JobClient::try_shutdown(std::uint64_t sequence, std::uint64_t session_id,
                                     std::uint64_t reason) noexcept
{
  return map_ring_result(requests_.try_push(make_shutdown_message(sequence, session_id, reason)));
}

ClientResult JobClient::map_ring_result(RingResult result) noexcept
{
  switch (result) {
  case RingResult::kSuccess:
    return ClientResult::kSuccess;
  case RingResult::kEmpty:
    return ClientResult::kEmpty;
  case RingResult::kFull:
    return ClientResult::kFull;
  case RingResult::kCorrupt:
    return ClientResult::kCorrupt;
  case RingResult::kTooLarge:
  case RingResult::kDestinationTooSmall:
    return ClientResult::kTransportError;
  }
  return ClientResult::kTransportError;
}

} // namespace fe::relay
