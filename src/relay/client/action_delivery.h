#pragma once

#include "relay/protocol/action_chunks.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string_view>

namespace fe::relay {

enum class UnsafeActionPolicy : std::uint8_t
{
  kReject,
  kClamp,
};

enum class ActionChunkAcceptResult : std::uint8_t
{
  kAccepted,
  kReplaced,
  kInvalid,
  kStale,
  kExpired,
  kSessionMismatch,
  kUnsafe,
};

enum class ActionStepError : std::uint8_t
{
  kEmpty,
  kNotReady,
  kExpired,
  kUnsafe,
  kInvalidTime,
};

enum class ActionDeliveryCreateError : std::uint8_t
{
  kInvalidModel,
  kInvalidConfig,
  kInvalidOverlap,
  kLimitShapeMismatch,
  kInvalidLimit,
};

[[nodiscard]] constexpr std::string_view to_string(ActionDeliveryCreateError error) noexcept
{
  switch (error) {
  case ActionDeliveryCreateError::kInvalidModel:
    return "invalid_model";
  case ActionDeliveryCreateError::kInvalidConfig:
    return "invalid_config";
  case ActionDeliveryCreateError::kInvalidOverlap:
    return "invalid_overlap";
  case ActionDeliveryCreateError::kLimitShapeMismatch:
    return "limit_shape_mismatch";
  case ActionDeliveryCreateError::kInvalidLimit:
    return "invalid_limit";
  }
  return "unknown";
}

struct ActionDeliveryConfig
{
  std::size_t control_dim{};
  std::size_t overlap_steps{};
  std::uint64_t step_period_ns{};
  std::uint64_t max_source_age_ns{};
  UnsafeActionPolicy unsafe_policy{UnsafeActionPolicy::kReject};
};

struct ActionSafetyLimits
{
  std::span<const float> lower{};
  std::span<const float> upper{};
  std::span<const float> max_delta_per_step{};
};

struct ActionStepView
{
  std::span<const float> values{};
  std::uint64_t sequence{};
  std::uint64_t session_id{};
  std::uint64_t generation{};
  std::uint32_t step_index{};
  bool blended{};
  bool clamped{};
};

struct ActionDeliveryCounters
{
  std::uint64_t accepted{};
  std::uint64_t replaced{};
  std::uint64_t rejected_invalid{};
  std::uint64_t rejected_stale{};
  std::uint64_t rejected_expired{};
  std::uint64_t rejected_session{};
  std::uint64_t rejected_unsafe{};
  std::uint64_t published{};
  std::uint64_t blended{};
  std::uint64_t clamped_values{};
  std::uint64_t skipped_steps{};
};

// Controller-side single-session gate. Construction copies limits once;
// acceptance, replacement, and step publication allocate nothing.
class ActionDeliveryGate
{
public:
  [[nodiscard]] static std::expected<ActionDeliveryGate, ActionDeliveryCreateError> create(
      const fe_model_metadata& model, ActionDeliveryConfig config,
      ActionSafetyLimits limits) noexcept;

  [[nodiscard]] ActionChunkAcceptResult accept(const ActionMessage& action, std::uint64_t now_ns,
                                               std::uint64_t valid_from_ns = 0u) noexcept;
  [[nodiscard]] ActionChunkAcceptResult accept(const ActionChunkView& chunk,
                                               std::uint64_t now_ns) noexcept;
  [[nodiscard]] std::expected<ActionStepView, ActionStepError> next(std::uint64_t now_ns) noexcept;

  // Reset changes session ownership but intentionally preserves counters.
  void reset() noexcept;
  [[nodiscard]] bool active() const noexcept { return active_; }
  [[nodiscard]] std::uint64_t newest_generation() const noexcept { return newest_generation_; }
  [[nodiscard]] const ActionDeliveryCounters& counters() const noexcept { return counters_; }

private:
  ActionDeliveryGate(const fe_model_metadata& model, ActionDeliveryConfig config) noexcept;
  [[nodiscard]] bool same_model(const fe_model_digest& digest) const noexcept;
  [[nodiscard]] bool within_absolute_limits(std::span<const float> values) const noexcept;
  [[nodiscard]] ActionChunkAcceptResult accept_impl(const ActionChunkView& chunk,
                                                    std::uint64_t now_ns,
                                                    bool values_validated) noexcept;
  void capture_overlap(const ActionChunkView& replacement) noexcept;

  ActionDeliveryConfig config_{};
  fe_model_digest model_digest_{};
  std::size_t model_action_values_{};
  std::array<float, kMaxActionDim> lower_{};
  std::array<float, kMaxActionDim> upper_{};
  std::array<float, kMaxActionDim> max_delta_{};
  std::array<float, kMaxActionDim> active_values_{};
  std::array<float, kMaxActionDim> overlap_values_{};
  std::array<float, kMaxActionDim> output_{};
  std::array<float, kMaxActionDim> pending_output_{};
  ActionChunkDescriptor active_chunk_{};
  ActionDeliveryCounters counters_{};
  std::size_t next_step_{};
  std::size_t overlap_steps_{};
  std::uint64_t session_id_{};
  std::uint64_t newest_generation_{};
  std::uint64_t newest_sequence_{};
  std::uint64_t last_publish_ns_{};
  bool active_{};
  bool session_bound_{};
  bool has_last_output_{};
};

} // namespace fe::relay
