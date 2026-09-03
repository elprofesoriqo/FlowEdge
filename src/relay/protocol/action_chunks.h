#pragma once

#include "relay/protocol/messages.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <span>
#include <string_view>
#include <type_traits>

namespace fe::relay {

inline constexpr std::uint32_t kActionChunkProtocolVersion = 1u;

enum class ActionChunkError : std::uint8_t
{
  kInvalidAction,
  kIncompleteAction,
  kInvalidShape,
  kInvalidTime,
  kExpired,
  kNonFinite,
};

struct ActionChunkDescriptor
{
  std::uint32_t struct_size{sizeof(ActionChunkDescriptor)};
  std::uint32_t protocol_version{kActionChunkProtocolVersion};
  std::uint32_t control_dim{};
  std::uint32_t step_count{};
  fe_model_digest model_digest{};
  std::uint64_t sequence{};
  std::uint64_t session_id{};
  std::uint64_t generation{};
  std::uint64_t source_timestamp_ns{};
  std::uint64_t source_deadline_ns{};
  std::uint64_t produced_ns{};
  std::uint64_t valid_from_ns{};
  std::uint64_t valid_until_ns{};
};
static_assert(sizeof(ActionChunkDescriptor) == 96uz);
static_assert(std::is_trivially_copyable_v<ActionChunkDescriptor>);

struct ActionChunkView
{
  ActionChunkDescriptor descriptor{};
  std::span<const float> values{};

  [[nodiscard]] std::span<const float> step(std::size_t index) const noexcept
  {
    const std::size_t width = descriptor.control_dim;
    if (index >= descriptor.step_count || width == 0uz || index > values.size() / width ||
        width > values.size() - (index * width))
      return {};
    return values.subspan(index * width, width);
  }
};

[[nodiscard]] constexpr std::string_view to_string(ActionChunkError error) noexcept
{
  switch (error) {
  case ActionChunkError::kInvalidAction:
    return "invalid_action";
  case ActionChunkError::kIncompleteAction:
    return "incomplete_action";
  case ActionChunkError::kInvalidShape:
    return "invalid_shape";
  case ActionChunkError::kInvalidTime:
    return "invalid_time";
  case ActionChunkError::kExpired:
    return "expired";
  case ActionChunkError::kNonFinite:
    return "non_finite";
  }
  return "unknown";
}

[[nodiscard]] inline std::expected<ActionChunkView, ActionChunkError> make_action_chunk(
    const ActionMessage& action, std::size_t control_dim, std::uint64_t produced_ns,
    std::uint64_t valid_from_ns, std::uint64_t step_period_ns) noexcept
{
  if (validate(action) != ProtocolResult::kSuccess)
    return std::unexpected(ActionChunkError::kInvalidAction);
  if (action_code(action) != RelayActionCode::kInference ||
      action.metadata.status != FE_ACTION_COMPLETE || action.metadata.remaining_nfe != 0u)
    return std::unexpected(ActionChunkError::kIncompleteAction);
  if (control_dim == 0uz || control_dim > kMaxActionDim || action.metadata.action_dim == 0u ||
      action.metadata.action_dim % control_dim != 0u)
    return std::unexpected(ActionChunkError::kInvalidShape);

  const std::uint64_t step_count = action.metadata.action_dim / control_dim;
  if (step_count == 0u || step_count > std::numeric_limits<std::uint32_t>::max() ||
      step_period_ns == 0u)
    return std::unexpected(ActionChunkError::kInvalidShape);
  if ((action.metadata.timestamp_ns != 0u && produced_ns < action.metadata.timestamp_ns) ||
      (action.metadata.deadline_ns != 0u &&
       action.metadata.deadline_ns < action.metadata.timestamp_ns))
    return std::unexpected(ActionChunkError::kInvalidTime);
  if (action.metadata.deadline_ns != 0u && produced_ns > action.metadata.deadline_ns)
    return std::unexpected(ActionChunkError::kExpired);

  const std::uint64_t start = valid_from_ns == 0u ? produced_ns : valid_from_ns;
  if (step_count > (std::numeric_limits<std::uint64_t>::max() - start) / step_period_ns)
    return std::unexpected(ActionChunkError::kInvalidTime);
  const std::uint64_t valid_until = start + (step_count * step_period_ns);
  if (valid_until <= produced_ns)
    return std::unexpected(ActionChunkError::kExpired);

  const std::span<const float> values{action.action.data(),
                                      static_cast<std::size_t>(action.metadata.action_dim)};
  for (const float value : values)
    if (!std::isfinite(value))
      return std::unexpected(ActionChunkError::kNonFinite);

  ActionChunkDescriptor descriptor{};
  descriptor.control_dim = static_cast<std::uint32_t>(control_dim);
  descriptor.step_count = static_cast<std::uint32_t>(step_count);
  descriptor.model_digest = action.metadata.model_digest;
  descriptor.sequence = action.envelope.sequence;
  descriptor.session_id = action.envelope.session_id;
  descriptor.generation = action.metadata.generation;
  descriptor.source_timestamp_ns = action.metadata.timestamp_ns;
  descriptor.source_deadline_ns = action.metadata.deadline_ns;
  descriptor.produced_ns = produced_ns;
  descriptor.valid_from_ns = start;
  descriptor.valid_until_ns = valid_until;
  return ActionChunkView{.descriptor = descriptor, .values = values};
}

} // namespace fe::relay
