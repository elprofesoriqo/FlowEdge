#include "relay/client/relay_client.h"

#include <utility>

namespace fe::relay {
namespace {

[[nodiscard]] bool usable_ring(const SharedMemoryRing& ring, std::size_t slot_bytes) noexcept
{
  const RingConfig config = ring.config();
  return config.capacity >= 2u && config.slot_bytes >= slot_bytes;
}

} // namespace

std::expected<void, std::string> RelayClient::validate_model(
    const fe_model_metadata& model) noexcept
{
  if (model.struct_size < sizeof(fe_model_metadata) ||
      model.protocol_version != FE_PROTOCOL_VERSION || model.condition_dim == 0u ||
      model.action_dim == 0u || model.condition_dim > kMaxConditionDim ||
      model.action_dim > kMaxActionDim)
    return std::unexpected("Relay client requires compatible flow-head model metadata");
  return {};
}

std::expected<RelayClient, std::string> RelayClient::create(std::string condition_name,
                                                            std::string action_name,
                                                            const fe_model_metadata& model,
                                                            std::uint32_t capacity) noexcept
{
  const auto valid_model = validate_model(model);
  if (!valid_model)
    return std::unexpected(valid_model.error());
  auto conditions = SharedMemoryRing::create(std::move(condition_name),
                                             RingConfig{.capacity = capacity,
                                                        .slot_bytes = sizeof(ConditionMessage)});
  if (!conditions)
    return std::unexpected(conditions.error());
  auto actions = SharedMemoryRing::create(std::move(action_name),
                                          RingConfig{.capacity = capacity,
                                                     .slot_bytes = sizeof(ActionMessage)});
  if (!actions)
    return std::unexpected(actions.error());
  return RelayClient{std::move(*conditions), std::move(*actions), model};
}

std::expected<RelayClient, std::string> RelayClient::connect(
    std::string condition_name, std::string action_name, const fe_model_metadata& model) noexcept
{
  const auto valid_model = validate_model(model);
  if (!valid_model)
    return std::unexpected(valid_model.error());
  auto conditions = SharedMemoryRing::open(std::move(condition_name));
  if (!conditions)
    return std::unexpected(conditions.error());
  auto actions = SharedMemoryRing::open(std::move(action_name));
  if (!actions)
    return std::unexpected(actions.error());
  if (!usable_ring(*conditions, sizeof(ConditionMessage)) ||
      !usable_ring(*actions, sizeof(ActionMessage)))
    return std::unexpected("Relay ring slot capacity is incompatible with this client build");
  return RelayClient{std::move(*conditions), std::move(*actions), model};
}

std::expected<ClientResult, std::string> RelayClient::send_shutdown(std::string condition_name,
                                                                    std::uint64_t sequence,
                                                                    std::uint64_t session_id,
                                                                    std::uint64_t reason) noexcept
{
  auto conditions = SharedMemoryRing::open(std::move(condition_name));
  if (!conditions)
    return std::unexpected(conditions.error());
  if (!usable_ring(*conditions, sizeof(ControlMessage)))
    return std::unexpected("Relay condition ring cannot hold a control message");
  const ControlMessage shutdown = make_shutdown_message(sequence, session_id, reason);
  return map_ring_result(conditions->try_push(shutdown));
}

ClientResult RelayClient::try_submit(const RelayRequest& request) noexcept
{
  if (make_condition_message(outbound_, model_, request.sequence, request.session_id,
                             request.timestamp_ns, request.deadline_ns, request.generation,
                             request.solver_steps, request.solver, request.condition,
                             request.noise) != ProtocolResult::kSuccess)
    return ClientResult::kInvalidRequest;
  return map_ring_result(conditions_.try_push(wire_bytes(outbound_)));
}

ClientResult RelayClient::try_receive(ActionMessage& action) noexcept
{
  action = {};
  std::size_t bytes_read{};
  const RingResult result =
      actions_.try_pop(std::as_writable_bytes(std::span{&action, 1uz}), bytes_read);
  if (result != RingResult::kSuccess)
    return map_ring_result(result);
  if (bytes_read != wire_size(action) || validate(action) != ProtocolResult::kSuccess)
    return ClientResult::kCorrupt;
  if (!compatible(action, model_))
    return ClientResult::kIncompatible;
  return ClientResult::kSuccess;
}

ClientResult RelayClient::try_shutdown(std::uint64_t sequence, std::uint64_t session_id,
                                       std::uint64_t reason) noexcept
{
  const ControlMessage shutdown = make_shutdown_message(sequence, session_id, reason);
  return map_ring_result(conditions_.try_push(shutdown));
}

ClientResult RelayClient::map_ring_result(RingResult result) noexcept
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

std::string_view to_string(ClientResult result) noexcept
{
  switch (result) {
  case ClientResult::kSuccess:
    return "success";
  case ClientResult::kEmpty:
    return "empty";
  case ClientResult::kFull:
    return "full";
  case ClientResult::kInvalidRequest:
    return "invalid request";
  case ClientResult::kIncompatible:
    return "incompatible model";
  case ClientResult::kCorrupt:
    return "corrupt message";
  case ClientResult::kTransportError:
    return "transport error";
  }
  return "unknown";
}

} // namespace fe::relay
