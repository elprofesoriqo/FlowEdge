#pragma once

#include "protocol/contracts.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

#ifndef FLOWEDGE_RELAY_MAX_CONDITION_DIM
#define FLOWEDGE_RELAY_MAX_CONDITION_DIM 4096
#endif

#ifndef FLOWEDGE_RELAY_MAX_ACTION_DIM
#define FLOWEDGE_RELAY_MAX_ACTION_DIM 1024
#endif

namespace fe::relay {

inline constexpr std::uint32_t kMessageMagic = 0x31455246u; // "FRE1" little-endian
inline constexpr std::uint16_t kMessageVersion = 1u;
inline constexpr std::size_t kMaxConditionDim = FLOWEDGE_RELAY_MAX_CONDITION_DIM;
inline constexpr std::size_t kMaxActionDim = FLOWEDGE_RELAY_MAX_ACTION_DIM;

enum class MessageKind : std::uint16_t
{
  kUnknown = 0,
  kCondition = 1,
  kAction = 2,
  kShutdown = 3,
};

struct MessageEnvelope
{
  std::uint32_t magic{kMessageMagic};
  std::uint16_t version{kMessageVersion};
  MessageKind kind{};
  std::uint32_t struct_size{};
  std::uint32_t flags{};
  std::uint64_t sequence{};
  std::uint64_t session_id{};
};
static_assert(sizeof(MessageEnvelope) == 32);

struct ConditionMessage
{
  MessageEnvelope envelope{.kind = MessageKind::kCondition};
  fe_condition_metadata metadata{};
  std::array<float, kMaxConditionDim + kMaxActionDim> payload{};

  [[nodiscard]] std::span<float> condition_values() noexcept
  {
    const auto count =
        static_cast<std::size_t>(std::min<std::uint64_t>(metadata.condition_dim, kMaxConditionDim));
    return {payload.data(), count};
  }
  [[nodiscard]] std::span<const float> condition_values() const noexcept
  {
    const auto count =
        static_cast<std::size_t>(std::min<std::uint64_t>(metadata.condition_dim, kMaxConditionDim));
    return {payload.data(), count};
  }
  [[nodiscard]] std::span<float> noise_values() noexcept
  {
    const auto condition =
        static_cast<std::size_t>(std::min<std::uint64_t>(metadata.condition_dim, kMaxConditionDim));
    const auto action =
        static_cast<std::size_t>(std::min<std::uint64_t>(metadata.action_dim, kMaxActionDim));
    return {payload.data() + condition, action};
  }
  [[nodiscard]] std::span<const float> noise_values() const noexcept
  {
    const auto condition =
        static_cast<std::size_t>(std::min<std::uint64_t>(metadata.condition_dim, kMaxConditionDim));
    const auto action =
        static_cast<std::size_t>(std::min<std::uint64_t>(metadata.action_dim, kMaxActionDim));
    return {payload.data() + condition, action};
  }
};

struct ActionMessage
{
  MessageEnvelope envelope{.kind = MessageKind::kAction};
  fe_action_metadata metadata{};
  std::array<float, kMaxActionDim> action{};
};

struct ControlMessage
{
  MessageEnvelope envelope{.kind = MessageKind::kShutdown};
  std::uint64_t reason{};
};

[[nodiscard]] constexpr ControlMessage make_shutdown_message(std::uint64_t sequence,
                                                             std::uint64_t session_id,
                                                             std::uint64_t reason = 0u) noexcept
{
  ControlMessage message{};
  message.envelope.struct_size = sizeof(message);
  message.envelope.sequence = sequence;
  message.envelope.session_id = session_id;
  message.reason = reason;
  return message;
}

[[nodiscard]] constexpr std::size_t wire_size(const ConditionMessage& message) noexcept
{
  const std::size_t condition = static_cast<std::size_t>(
      std::min<std::uint64_t>(message.metadata.condition_dim, kMaxConditionDim));
  const std::size_t action =
      static_cast<std::size_t>(std::min<std::uint64_t>(message.metadata.action_dim, kMaxActionDim));
  return offsetof(ConditionMessage, payload) + ((condition + action) * sizeof(float));
}

[[nodiscard]] constexpr std::size_t wire_size(const ActionMessage& message) noexcept
{
  const std::size_t action =
      static_cast<std::size_t>(std::min<std::uint64_t>(message.metadata.action_dim, kMaxActionDim));
  return offsetof(ActionMessage, action) + (action * sizeof(float));
}

[[nodiscard]] inline std::span<const std::byte> wire_bytes(const ConditionMessage& message) noexcept
{
  return std::as_bytes(std::span{&message, 1uz}).first(wire_size(message));
}

[[nodiscard]] inline std::span<const std::byte> wire_bytes(const ActionMessage& message) noexcept
{
  return std::as_bytes(std::span{&message, 1uz}).first(wire_size(message));
}

enum class ProtocolResult : std::uint8_t
{
  kSuccess,
  kInvalidEnvelope,
  kDimensionExceeded,
  kInvalidMetadata,
};

[[nodiscard]] constexpr std::uint64_t required_nfe(std::uint64_t steps,
                                                   std::uint32_t solver) noexcept
{
  const std::uint64_t per_step = solver == 2u ? 4u : solver == 1u ? 2u : 1u;
  return steps > (std::numeric_limits<std::uint64_t>::max() / per_step)
             ? std::numeric_limits<std::uint64_t>::max()
             : steps * per_step;
}

[[nodiscard]] constexpr bool valid_envelope(const MessageEnvelope& envelope, MessageKind expected,
                                            std::size_t expected_size) noexcept
{
  return envelope.magic == kMessageMagic && envelope.version == kMessageVersion &&
         envelope.kind == expected && envelope.struct_size == expected_size && envelope.flags == 0u;
}

[[nodiscard]] inline ProtocolResult validate(const ConditionMessage& message) noexcept
{
  if (message.envelope.magic != kMessageMagic || message.envelope.version != kMessageVersion ||
      message.envelope.kind != MessageKind::kCondition || message.envelope.flags != 0u)
    return ProtocolResult::kInvalidEnvelope;
  if (message.metadata.condition_dim > kMaxConditionDim ||
      message.metadata.action_dim > kMaxActionDim)
    return ProtocolResult::kDimensionExceeded;
  if (message.envelope.struct_size != wire_size(message))
    return ProtocolResult::kInvalidEnvelope;
  if (message.metadata.struct_size < sizeof(fe_condition_metadata) ||
      message.metadata.protocol_version != FE_PROTOCOL_VERSION || message.metadata.reserved != 0u ||
      message.metadata.solver > 2u || message.metadata.solver_steps == 0u ||
      message.metadata.remaining_nfe !=
          required_nfe(message.metadata.solver_steps, message.metadata.solver) ||
      (message.metadata.deadline_ns != 0u &&
       message.metadata.deadline_ns < message.metadata.timestamp_ns))
    return ProtocolResult::kInvalidMetadata;
  return ProtocolResult::kSuccess;
}

[[nodiscard]] inline ProtocolResult validate(const ActionMessage& message) noexcept
{
  if (message.envelope.magic != kMessageMagic || message.envelope.version != kMessageVersion ||
      message.envelope.kind != MessageKind::kAction || message.envelope.flags != 0u)
    return ProtocolResult::kInvalidEnvelope;
  if (message.metadata.condition_dim > kMaxConditionDim ||
      message.metadata.action_dim > kMaxActionDim)
    return ProtocolResult::kDimensionExceeded;
  if (message.envelope.struct_size != wire_size(message))
    return ProtocolResult::kInvalidEnvelope;
  if (message.metadata.struct_size < sizeof(fe_action_metadata) ||
      message.metadata.protocol_version != FE_PROTOCOL_VERSION || message.metadata.solver > 2u ||
      message.metadata.status > FE_ACTION_FAILED)
    return ProtocolResult::kInvalidMetadata;
  return ProtocolResult::kSuccess;
}

[[nodiscard]] inline bool compatible(const ConditionMessage& message,
                                     const fe_model_metadata& model) noexcept
{
  return validate(message) == ProtocolResult::kSuccess &&
         model.protocol_version == FE_PROTOCOL_VERSION &&
         message.metadata.condition_dim == model.condition_dim &&
         message.metadata.action_dim == model.action_dim &&
         std::ranges::equal(message.metadata.model_digest.bytes, model.model_digest.bytes);
}

[[nodiscard]] inline ProtocolResult make_condition_message(
    ConditionMessage& destination, const fe_model_metadata& model, std::uint64_t sequence,
    std::uint64_t session_id, std::uint64_t timestamp_ns, std::uint64_t deadline_ns,
    std::uint64_t generation, std::size_t steps, std::uint32_t solver,
    std::span<const float> condition, std::span<const float> noise) noexcept
{
  if (model.protocol_version != FE_PROTOCOL_VERSION || steps == 0uz || solver > 2u ||
      (deadline_ns != 0u && deadline_ns < timestamp_ns))
    return ProtocolResult::kInvalidMetadata;
  if (condition.size() != model.condition_dim || noise.size() != model.action_dim ||
      condition.size() > kMaxConditionDim || noise.size() > kMaxActionDim)
    return ProtocolResult::kDimensionExceeded;

  destination.envelope = {};
  destination.metadata = {};
  destination.envelope.kind = MessageKind::kCondition;
  destination.envelope.sequence = sequence;
  destination.envelope.session_id = session_id;
  destination.metadata.struct_size = sizeof(fe_condition_metadata);
  destination.metadata.protocol_version = FE_PROTOCOL_VERSION;
  destination.metadata.solver = solver;
  destination.metadata.model_digest = model.model_digest;
  destination.metadata.timestamp_ns = timestamp_ns;
  destination.metadata.deadline_ns = deadline_ns;
  destination.metadata.generation = generation;
  destination.metadata.condition_dim = model.condition_dim;
  destination.metadata.action_dim = model.action_dim;
  destination.metadata.solver_steps = steps;
  destination.metadata.remaining_nfe = required_nfe(steps, solver);
  destination.envelope.struct_size = static_cast<std::uint32_t>(wire_size(destination));
  std::ranges::copy(condition, destination.condition_values().begin());
  std::ranges::copy(noise, destination.noise_values().begin());
  return ProtocolResult::kSuccess;
}

static_assert(std::is_trivially_copyable_v<MessageEnvelope>);
static_assert(std::is_trivially_copyable_v<ConditionMessage>);
static_assert(std::is_trivially_copyable_v<ActionMessage>);
static_assert(std::is_trivially_copyable_v<ControlMessage>);

} // namespace fe::relay
