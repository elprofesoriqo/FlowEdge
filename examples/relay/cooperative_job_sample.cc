#include "relay/adapters/cooperative_adapters.h"
#include "relay/jobs/state_codec.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

namespace {

using namespace fe::relay;

struct IterativeBackend
{
  static constexpr std::uint64_t kSteps = 8u;

  [[nodiscard]] bool begin() noexcept
  {
    step = 0u;
    latent = {1.0f, -1.0f, 0.5f, -0.5f};
    cancelled = false;
    return true;
  }

  [[nodiscard]] BackendAdvance advance(std::size_t budget) noexcept
  {
    const std::size_t count =
        static_cast<std::size_t>(std::min<std::uint64_t>(budget, kSteps - step));
    for (std::size_t unit{}; unit < count; ++unit) {
      ++step;
      for (float& value : latent)
        value = (value * 0.875f) + static_cast<float>(step) * 0.015625f;
    }
    return {
        .step = step == kSteps ? BackendStep::kComplete : BackendStep::kInProgress,
        .completed_work_units = count,
    };
  }

  void cancel() noexcept { cancelled = true; }
  [[nodiscard]] std::size_t state_bytes() const noexcept { return 25uz; }

  [[nodiscard]] bool save_state(std::span<std::byte> destination) const noexcept
  {
    StateWriter writer{destination};
    if (!writer.write(step))
      return false;
    for (const float value : latent) {
      if (!writer.write_float(value))
        return false;
    }
    return writer.write(static_cast<std::uint8_t>(cancelled)) && writer.remaining() == 0uz;
  }

  [[nodiscard]] bool load_state(std::span<const std::byte> source) noexcept
  {
    StateReader reader{source};
    const auto next_step = reader.read<std::uint64_t>();
    std::array<float, 4> next_latent{};
    for (float& value : next_latent) {
      const auto decoded = reader.read_float();
      if (!decoded)
        return false;
      value = *decoded;
    }
    const auto next_cancelled = reader.read<std::uint8_t>();
    if (!next_step || !next_cancelled || *next_step > kSteps || *next_cancelled > 1u ||
        reader.remaining() != 0uz)
      return false;
    step = *next_step;
    latent = next_latent;
    cancelled = *next_cancelled != 0u;
    return true;
  }

  std::uint64_t step{};
  std::array<float, 4> latent{};
  bool cancelled{};
};

[[nodiscard]] JobDescriptor descriptor(std::uint64_t generation = 1u)
{
  JobDescriptor result{};
  for (std::size_t index{}; index < result.model_digest.size(); ++index)
    result.model_digest[index] = static_cast<std::uint8_t>(index + 1uz);
  result.state_schema = 0x697465722d7631u; // "iter-v1"
  result.session_id = 9001u;
  result.generation = generation;
  result.deadline_ns = 1'000'000'000u;
  result.total_work_units = IterativeBackend::kSteps;
  return result;
}

} // namespace

int main()
{
  IterativeBackend source_backend{};
  auto source_result = make_iterative_job(source_backend, descriptor());
  if (!source_result)
    return 1;
  CooperativeJob source = *source_result;
  if (!source.start() || !source.advance(3uz))
    return 1;

  std::vector<std::byte> capsule(source.capsule_bytes());
  if (!source.export_capsule(capsule))
    return 1;

  IterativeBackend migrated_backend{};
  auto migrated_result = make_iterative_job(migrated_backend, descriptor());
  if (!migrated_result)
    return 1;
  CooperativeJob migrated = *migrated_result;
  const auto restored = migrated.restore_capsule(capsule);
  const auto complete =
      restored ? migrated.advance(IterativeBackend::kSteps)
               : std::expected<JobProgress, JobError>{std::unexpected(restored.error())};
  if (!complete || complete->state != JobState::kComplete)
    return 1;

  IterativeBackend streaming_backend{};
  auto streaming = make_streaming_job(streaming_backend, descriptor(4u));
  if (!streaming || !streaming->start() || !streaming->advance(1uz))
    return 1;
  const JobProgress cancelled = streaming->cancel_before(5u);

  IterativeBackend speculative_backend{};
  auto speculative = make_speculative_job(speculative_backend, descriptor(6u));
  if (!speculative || !speculative->start() || !speculative->advance(8uz))
    return 1;

  std::cout << "migrated_kind=iterative completed=" << complete->completed_work_units
            << " capsule_bytes=" << capsule.size() << " latent0=" << migrated_backend.latent[0]
            << '\n';
  std::cout << "streaming_cancelled=" << (cancelled.state == JobState::kCancelled)
            << " speculative_complete=" << (speculative->progress().state == JobState::kComplete)
            << '\n';
  return 0;
}
