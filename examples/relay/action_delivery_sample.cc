#include "relay/client/action_delivery.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <ranges>
#include <span>

namespace {

using namespace fe::relay;

[[nodiscard]] fe_model_metadata model_metadata() noexcept
{
  fe_model_metadata model{};
  model.struct_size = sizeof(model);
  model.protocol_version = FE_PROTOCOL_VERSION;
  model.architecture = FE_ARCH_FLOW_HEAD;
  model.precision = FE_PRECISION_F32;
  model.condition_dim = 4u;
  model.action_dim = 8u;
  for (std::size_t index{}; index < FE_MODEL_DIGEST_BYTES; ++index)
    model.model_digest.bytes[index] = static_cast<std::uint8_t>(index + 1uz);
  return model;
}

[[nodiscard]] ActionMessage action_message(const fe_model_metadata& model,
                                           std::span<const float> values, std::uint64_t sequence,
                                           std::uint64_t generation, std::uint64_t timestamp_ns,
                                           std::uint64_t deadline_ns = 0u) noexcept
{
  ActionMessage action{};
  action.envelope.kind = MessageKind::kAction;
  action.envelope.sequence = sequence;
  action.envelope.session_id = 42u;
  action.metadata.struct_size = sizeof(fe_action_metadata);
  action.metadata.protocol_version = FE_PROTOCOL_VERSION;
  action.metadata.solver = 0u;
  action.metadata.status = FE_ACTION_COMPLETE;
  action.metadata.model_digest = model.model_digest;
  action.metadata.timestamp_ns = timestamp_ns;
  action.metadata.deadline_ns = deadline_ns;
  action.metadata.generation = generation;
  action.metadata.condition_dim = model.condition_dim;
  action.metadata.action_dim = model.action_dim;
  std::ranges::copy(values, action.action.begin());
  action.envelope.struct_size = static_cast<std::uint32_t>(wire_size(action));
  return action;
}

} // namespace

int main()
{
  using namespace fe::relay;
  constexpr std::uint64_t kPeriodNs = 10'000'000u;
  constexpr std::uint64_t kStartNs = 1'000'000'000u;
  constexpr std::array lower{-1.0F, -1.0F};
  constexpr std::array upper{1.0F, 1.0F};
  constexpr std::array max_delta{0.25F, 0.25F};
  const fe_model_metadata model = model_metadata();
  auto created =
      ActionDeliveryGate::create(model,
                                 ActionDeliveryConfig{.control_dim = 2uz,
                                                      .overlap_steps = 2uz,
                                                      .step_period_ns = kPeriodNs,
                                                      .max_source_age_ns = 80'000'000u,
                                                      .unsafe_policy = UnsafeActionPolicy::kClamp},
                                 ActionSafetyLimits{.lower = lower,
                                                    .upper = upper,
                                                    .max_delta_per_step = max_delta});
  if (!created) {
    std::cerr << to_string(created.error()) << '\n';
    return 1;
  }
  ActionDeliveryGate gate = std::move(*created);

  constexpr std::array first_values{0.0F, 0.0F, 0.2F, 0.1F, 0.4F, 0.2F, 0.6F, 0.3F};
  const ActionMessage first = action_message(model, first_values, 1u, 1u, kStartNs);
  if (gate.accept(first, kStartNs, kStartNs) != ActionChunkAcceptResult::kAccepted)
    return 1;

  std::array<float, 2> last{};
  for (std::size_t tick{}; tick < 6uz; ++tick) {
    const std::uint64_t now = kStartNs + (tick * kPeriodNs);
    if (tick == 2uz) {
      constexpr std::array replacement_values{0.9F, -0.9F, 0.8F, -0.8F, 0.4F, -0.4F, 0.0F, 0.0F};
      const ActionMessage replacement = action_message(model, replacement_values, 2u, 2u, now);
      if (gate.accept(replacement, now, now) != ActionChunkAcceptResult::kReplaced)
        return 1;
    }
    const auto step = gate.next(now);
    if (!step)
      return 1;
    std::ranges::copy(step->values, last.begin());
  }

  const bool stale_rejected =
      gate.accept(first, kStartNs + (6u * kPeriodNs)) == ActionChunkAcceptResult::kStale;
  const ActionMessage expired =
      action_message(model, first_values, 3u, 3u, kStartNs, kStartNs + (3u * kPeriodNs));
  const bool expired_rejected =
      gate.accept(expired, kStartNs + (6u * kPeriodNs)) == ActionChunkAcceptResult::kExpired;
  const ActionDeliveryCounters& counters = gate.counters();
  std::cout << "policy_hz=25 controller_hz=100 published=" << counters.published
            << " replacements=" << counters.replaced << " blended=" << counters.blended
            << " clamped_values=" << counters.clamped_values << " stale_rejected=" << stale_rejected
            << " expired_rejected=" << expired_rejected << " final_action=" << last[0] << ','
            << last[1] << '\n';
  return counters.accepted == 2u && counters.published == 6u && counters.replaced == 1u &&
                 counters.blended == 2u && counters.clamped_values != 0u && stale_rejected &&
                 expired_rejected
             ? 0
             : 1;
}
