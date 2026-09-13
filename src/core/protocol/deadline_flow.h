#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <span>

namespace fe {

// Plans describe complete, independently validated trajectories. A trace/precision
// identifier is a capability requirement, never a request to compile in the hot path.
struct ExecutionPlan
{
  std::uint32_t id{};
  std::uint32_t steps{};
  std::uint32_t solver{};
  std::uint32_t precision{};
  std::uint32_t trace_id{};
  std::uint64_t hardware_mask{1u};
  std::uint64_t latency_ns{};
  std::uint64_t memory_bytes{};
  double quality{};
};

struct PlanContext
{
  std::uint64_t slack_ns{};
  std::uint64_t buffered_actions{};
  std::uint64_t period_ns{};
  std::uint64_t reserve_ns{};
  std::uint64_t switch_ns{};
  std::uint64_t available_memory_bytes{std::numeric_limits<std::uint64_t>::max()};
  std::uint64_t hardware_mask{1u};
  double minimum_quality{};
  bool healthy{true};
};

enum class PlanError : std::uint8_t
{
  kInvalidPortfolio,
};

class DeadlineFlow
{
public:
  static constexpr std::size_t kMaxPlans = 32uz;

  [[nodiscard]] static std::expected<DeadlineFlow, PlanError> create(
      std::span<const ExecutionPlan> plans) noexcept
  {
    if (plans.empty() || plans.size() > kMaxPlans)
      return std::unexpected(PlanError::kInvalidPortfolio);
    DeadlineFlow result;
    for (const auto& plan : plans) {
      if (plan.steps == 0u || plan.latency_ns == 0u || plan.hardware_mask == 0u ||
          !std::isfinite(plan.quality) || plan.quality < 0.0 || plan.quality > 1.0)
        return std::unexpected(PlanError::kInvalidPortfolio);
      for (std::size_t j{}; j < result.count_; ++j)
        if (result.plans_[j].id == plan.id)
          return std::unexpected(PlanError::kInvalidPortfolio);
      result.plans_[result.count_] = plan;
      result.estimates_[result.count_++] = plan.latency_ns;
    }
    return result;
  }

  [[nodiscard]] std::optional<std::uint32_t> select(const PlanContext& context) const noexcept
  {
    if (!context.healthy || !std::isfinite(context.minimum_quality) ||
        context.minimum_quality < 0.0 || context.minimum_quality > 1.0)
      return {};
    constexpr auto max = std::numeric_limits<std::uint64_t>::max();
    const auto buffer_ns =
        context.period_ns != 0u && context.buffered_actions > max / context.period_ns
            ? max
            : context.buffered_actions * context.period_ns;
    // No queued action means the explicit startup/deadline slack is the budget.
    const auto budget =
        context.buffered_actions == 0u ? context.slack_ns : std::min(context.slack_ns, buffer_ns);
    if (budget <= context.reserve_ns)
      return {};
    std::optional<std::size_t> best;
    std::uint64_t best_cost{max};
    for (std::size_t i{}; i < count_; ++i) {
      const auto& plan = plans_[i];
      if ((plan.hardware_mask & context.hardware_mask) != plan.hardware_mask ||
          plan.memory_bytes > context.available_memory_bytes ||
          plan.quality < context.minimum_quality)
        continue;
      const auto switching = last_ && *last_ != plan.id ? context.switch_ns : 0u;
      const auto available = budget - context.reserve_ns;
      if (switching > available || estimates_[i] > available - switching)
        continue;
      const auto cost = estimates_[i] + switching;
      if (!best || plan.quality > plans_[*best].quality ||
          (plan.quality == plans_[*best].quality && cost < best_cost)) {
        best = i;
        best_cost = cost;
      }
    }
    return best ? std::optional{plans_[*best].id} : std::nullopt;
  }

  // One selector per execution lane. Caller records complete dispatch-to-ready time.
  [[nodiscard]] bool observe(std::uint32_t id, std::uint64_t latency_ns) noexcept
  {
    if (latency_ns == 0u)
      return false;
    for (std::size_t i{}; i < count_; ++i) {
      if (plans_[i].id != id)
        continue;
      estimates_[i] =
          std::max({plans_[i].latency_ns, latency_ns, estimates_[i] - estimates_[i] / 8u});
      last_ = id;
      return true;
    }
    return false;
  }

private:
  std::array<ExecutionPlan, kMaxPlans> plans_{};
  std::array<std::uint64_t, kMaxPlans> estimates_{};
  std::size_t count_{};
  std::optional<std::uint32_t> last_{};
};

} // namespace fe
