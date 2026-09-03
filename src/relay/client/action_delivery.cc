#include "relay/client/action_delivery.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>

namespace fe::relay {
namespace {

[[nodiscard]] ActionChunkAcceptResult map_chunk_error(ActionChunkError error) noexcept
{
  switch (error) {
  case ActionChunkError::kExpired:
    return ActionChunkAcceptResult::kExpired;
  case ActionChunkError::kNonFinite:
    return ActionChunkAcceptResult::kUnsafe;
  case ActionChunkError::kInvalidAction:
  case ActionChunkError::kIncompleteAction:
  case ActionChunkError::kInvalidShape:
  case ActionChunkError::kInvalidTime:
    return ActionChunkAcceptResult::kInvalid;
  }
  return ActionChunkAcceptResult::kInvalid;
}

[[nodiscard]] bool valid_model(const fe_model_metadata& model) noexcept
{
  return model.struct_size >= sizeof(fe_model_metadata) &&
         model.protocol_version == FE_PROTOCOL_VERSION && model.action_dim != 0u &&
         model.action_dim <= kMaxActionDim;
}

} // namespace

ActionDeliveryGate::ActionDeliveryGate(const fe_model_metadata& model,
                                       ActionDeliveryConfig config) noexcept
    : config_{config}, model_digest_{model.model_digest}, model_action_values_{model.action_dim}
{
}

std::expected<ActionDeliveryGate, ActionDeliveryCreateError> ActionDeliveryGate::create(
    const fe_model_metadata& model, ActionDeliveryConfig config, ActionSafetyLimits limits) noexcept
{
  if (!valid_model(model))
    return std::unexpected(ActionDeliveryCreateError::kInvalidModel);
  if (config.control_dim == 0uz || config.control_dim > kMaxActionDim ||
      model.action_dim % config.control_dim != 0u || config.step_period_ns == 0u)
    return std::unexpected(ActionDeliveryCreateError::kInvalidConfig);
  const std::size_t step_count = static_cast<std::size_t>(model.action_dim) / config.control_dim;
  if (config.overlap_steps > step_count)
    return std::unexpected(ActionDeliveryCreateError::kInvalidOverlap);
  if (limits.lower.size() != config.control_dim || limits.upper.size() != config.control_dim ||
      limits.max_delta_per_step.size() != config.control_dim)
    return std::unexpected(ActionDeliveryCreateError::kLimitShapeMismatch);

  ActionDeliveryGate gate{model, config};
  for (std::size_t index{}; index < config.control_dim; ++index) {
    const float lower = limits.lower[index];
    const float upper = limits.upper[index];
    const float delta = limits.max_delta_per_step[index];
    if (!std::isfinite(lower) || !std::isfinite(upper) || !std::isfinite(delta) || lower > upper ||
        delta < 0.0F)
      return std::unexpected(ActionDeliveryCreateError::kInvalidLimit);
    gate.lower_[index] = lower;
    gate.upper_[index] = upper;
    gate.max_delta_[index] = delta;
  }
  return gate;
}

bool ActionDeliveryGate::same_model(const fe_model_digest& digest) const noexcept
{
  return std::ranges::equal(model_digest_.bytes, digest.bytes);
}

bool ActionDeliveryGate::within_absolute_limits(std::span<const float> values) const noexcept
{
  for (std::size_t index{}; index < values.size(); ++index) {
    const std::size_t component = index % config_.control_dim;
    if (values[index] < lower_[component] || values[index] > upper_[component])
      return false;
  }
  return true;
}

ActionChunkAcceptResult ActionDeliveryGate::accept(const ActionMessage& action,
                                                   std::uint64_t now_ns,
                                                   std::uint64_t valid_from_ns) noexcept
{
  const auto chunk =
      make_action_chunk(action, config_.control_dim, now_ns, valid_from_ns, config_.step_period_ns);
  if (!chunk) {
    const ActionChunkAcceptResult result = map_chunk_error(chunk.error());
    if (result == ActionChunkAcceptResult::kExpired)
      ++counters_.rejected_expired;
    else if (result == ActionChunkAcceptResult::kUnsafe)
      ++counters_.rejected_unsafe;
    else
      ++counters_.rejected_invalid;
    return result;
  }
  return accept_impl(*chunk, now_ns, true);
}

void ActionDeliveryGate::capture_overlap(const ActionChunkView& replacement) noexcept
{
  overlap_steps_ = 0uz;
  if (!active_ || config_.overlap_steps == 0uz ||
      replacement.descriptor.valid_from_ns < active_chunk_.valid_from_ns)
    return;

  const std::uint64_t elapsed = replacement.descriptor.valid_from_ns - active_chunk_.valid_from_ns;
  const std::size_t timed_step = static_cast<std::size_t>(elapsed / config_.step_period_ns);
  const std::size_t old_step = std::max(next_step_, timed_step);
  if (old_step >= active_chunk_.step_count)
    return;
  overlap_steps_ = std::min({config_.overlap_steps,
                             static_cast<std::size_t>(active_chunk_.step_count) - old_step,
                             static_cast<std::size_t>(replacement.descriptor.step_count)});
  for (std::size_t step{}; step < overlap_steps_; ++step) {
    const std::size_t source = (old_step + step) * config_.control_dim;
    const std::size_t destination = step * config_.control_dim;
    std::ranges::copy_n(active_values_.begin() + static_cast<std::ptrdiff_t>(source),
                        static_cast<std::ptrdiff_t>(config_.control_dim),
                        overlap_values_.begin() + static_cast<std::ptrdiff_t>(destination));
  }
}

ActionChunkAcceptResult ActionDeliveryGate::accept(const ActionChunkView& chunk,
                                                   std::uint64_t now_ns) noexcept
{
  return accept_impl(chunk, now_ns, false);
}

ActionChunkAcceptResult ActionDeliveryGate::accept_impl(const ActionChunkView& chunk,
                                                        std::uint64_t now_ns,
                                                        bool values_validated) noexcept
{
  const ActionChunkDescriptor& descriptor = chunk.descriptor;
  if (descriptor.control_dim == 0u || descriptor.step_count == 0u ||
      descriptor.step_count > kMaxActionDim / descriptor.control_dim ||
      descriptor.step_count >
          (std::numeric_limits<std::uint64_t>::max() - descriptor.valid_from_ns) /
              config_.step_period_ns) {
    ++counters_.rejected_invalid;
    return ActionChunkAcceptResult::kInvalid;
  }
  const std::size_t expected_values =
      static_cast<std::size_t>(descriptor.control_dim) * descriptor.step_count;
  const std::uint64_t expected_valid_until =
      descriptor.valid_from_ns + (descriptor.step_count * config_.step_period_ns);
  if (descriptor.struct_size != sizeof(ActionChunkDescriptor) ||
      descriptor.protocol_version != kActionChunkProtocolVersion ||
      descriptor.control_dim != config_.control_dim || expected_values != model_action_values_ ||
      chunk.values.size() != expected_values || !same_model(descriptor.model_digest) ||
      descriptor.produced_ns > now_ns ||
      (descriptor.source_timestamp_ns != 0u &&
       descriptor.produced_ns < descriptor.source_timestamp_ns) ||
      descriptor.valid_from_ns >= descriptor.valid_until_ns ||
      descriptor.valid_until_ns != expected_valid_until ||
      (config_.max_source_age_ns != 0u && descriptor.source_timestamp_ns == 0u)) {
    ++counters_.rejected_invalid;
    return ActionChunkAcceptResult::kInvalid;
  }
  if (!values_validated) {
    for (const float value : chunk.values) {
      if (!std::isfinite(value)) {
        ++counters_.rejected_unsafe;
        return ActionChunkAcceptResult::kUnsafe;
      }
    }
  }
  if ((descriptor.source_timestamp_ns != 0u && now_ns < descriptor.source_timestamp_ns) ||
      (descriptor.source_deadline_ns != 0u &&
       descriptor.source_deadline_ns < descriptor.source_timestamp_ns)) {
    ++counters_.rejected_invalid;
    return ActionChunkAcceptResult::kInvalid;
  }
  const bool source_too_old = config_.max_source_age_ns != 0u &&
                              descriptor.source_timestamp_ns != 0u &&
                              now_ns - descriptor.source_timestamp_ns > config_.max_source_age_ns;
  if (now_ns >= descriptor.valid_until_ns || source_too_old ||
      (descriptor.source_deadline_ns != 0u && now_ns > descriptor.source_deadline_ns)) {
    ++counters_.rejected_expired;
    return ActionChunkAcceptResult::kExpired;
  }
  if (config_.unsafe_policy == UnsafeActionPolicy::kReject &&
      !within_absolute_limits(chunk.values)) {
    ++counters_.rejected_unsafe;
    return ActionChunkAcceptResult::kUnsafe;
  }
  if (session_bound_ && descriptor.session_id != session_id_) {
    ++counters_.rejected_session;
    return ActionChunkAcceptResult::kSessionMismatch;
  }
  if (session_bound_ &&
      (descriptor.generation <= newest_generation_ || descriptor.sequence <= newest_sequence_)) {
    ++counters_.rejected_stale;
    return ActionChunkAcceptResult::kStale;
  }

  const bool replacing = active_;
  capture_overlap(chunk);
  std::ranges::copy(chunk.values, active_values_.begin());
  active_chunk_ = descriptor;
  next_step_ = 0uz;
  session_id_ = descriptor.session_id;
  newest_generation_ = descriptor.generation;
  newest_sequence_ = descriptor.sequence;
  active_ = true;
  session_bound_ = true;
  ++counters_.accepted;
  if (replacing) {
    ++counters_.replaced;
    return ActionChunkAcceptResult::kReplaced;
  }
  return ActionChunkAcceptResult::kAccepted;
}

std::expected<ActionStepView, ActionStepError> ActionDeliveryGate::next(
    std::uint64_t now_ns) noexcept
{
  if (!active_)
    return std::unexpected(ActionStepError::kEmpty);
  if (has_last_output_ && now_ns < last_publish_ns_)
    return std::unexpected(ActionStepError::kInvalidTime);
  if (now_ns < active_chunk_.valid_from_ns)
    return std::unexpected(ActionStepError::kNotReady);
  if (active_chunk_.source_timestamp_ns != 0u && now_ns < active_chunk_.source_timestamp_ns)
    return std::unexpected(ActionStepError::kInvalidTime);
  const bool source_too_old =
      config_.max_source_age_ns != 0u && active_chunk_.source_timestamp_ns != 0u &&
      now_ns - active_chunk_.source_timestamp_ns > config_.max_source_age_ns;
  if (now_ns >= active_chunk_.valid_until_ns || source_too_old) {
    active_ = false;
    ++counters_.rejected_expired;
    return std::unexpected(ActionStepError::kExpired);
  }

  const std::size_t step =
      static_cast<std::size_t>((now_ns - active_chunk_.valid_from_ns) / config_.step_period_ns);
  if (step >= active_chunk_.step_count) {
    active_ = false;
    ++counters_.rejected_expired;
    return std::unexpected(ActionStepError::kExpired);
  }
  if (step < next_step_)
    return std::unexpected(ActionStepError::kNotReady);
  counters_.skipped_steps += step - next_step_;

  const bool blended = step < overlap_steps_;
  const float blend =
      blended ? static_cast<float>(step + 1uz) / static_cast<float>(overlap_steps_ + 1uz) : 1.0F;
  bool clamped{};
  std::size_t clamped_values{};
  const std::size_t offset = step * config_.control_dim;
  std::uint64_t elapsed_steps{1u};
  if (has_last_output_ && now_ns > last_publish_ns_)
    elapsed_steps =
        std::max<std::uint64_t>(1u, (now_ns - last_publish_ns_) / config_.step_period_ns);

  for (std::size_t component{}; component < config_.control_dim; ++component) {
    const float fresh = active_values_[offset + component];
    const float candidate =
        blended ? std::lerp(overlap_values_[offset + component], fresh, blend) : fresh;
    const double allowed_delta =
        static_cast<double>(max_delta_[component]) * static_cast<double>(elapsed_steps);
    const bool absolute_safe = candidate >= lower_[component] && candidate <= upper_[component];
    const bool delta_safe =
        !has_last_output_ || std::abs(static_cast<double>(candidate) -
                                      static_cast<double>(output_[component])) <= allowed_delta;
    if (config_.unsafe_policy == UnsafeActionPolicy::kReject && (!absolute_safe || !delta_safe)) {
      active_ = false;
      ++counters_.rejected_unsafe;
      return std::unexpected(ActionStepError::kUnsafe);
    }

    float safe = std::clamp(candidate, lower_[component], upper_[component]);
    if (has_last_output_) {
      const double delta_lower = static_cast<double>(output_[component]) - allowed_delta;
      const double delta_upper = static_cast<double>(output_[component]) + allowed_delta;
      safe = static_cast<float>(std::clamp(static_cast<double>(safe), delta_lower, delta_upper));
      safe = std::clamp(safe, lower_[component], upper_[component]);
    }
    pending_output_[component] = safe;
    if (safe != candidate) {
      clamped = true;
      ++clamped_values;
    }
  }

  std::ranges::copy_n(pending_output_.begin(), static_cast<std::ptrdiff_t>(config_.control_dim),
                      output_.begin());
  counters_.clamped_values += clamped_values;

  next_step_ = step + 1uz;
  last_publish_ns_ = now_ns;
  has_last_output_ = true;
  ++counters_.published;
  if (blended)
    ++counters_.blended;
  if (next_step_ == active_chunk_.step_count)
    active_ = false;
  return ActionStepView{.values = std::span{output_}.first(config_.control_dim),
                        .sequence = active_chunk_.sequence,
                        .session_id = active_chunk_.session_id,
                        .generation = active_chunk_.generation,
                        .step_index = static_cast<std::uint32_t>(step),
                        .blended = blended,
                        .clamped = clamped};
}

void ActionDeliveryGate::reset() noexcept
{
  active_ = false;
  session_bound_ = false;
  has_last_output_ = false;
  next_step_ = 0uz;
  overlap_steps_ = 0uz;
  session_id_ = 0u;
  newest_generation_ = 0u;
  newest_sequence_ = 0u;
  last_publish_ns_ = 0u;
}

} // namespace fe::relay
