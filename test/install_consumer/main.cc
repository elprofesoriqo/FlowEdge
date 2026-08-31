#include "api/engine.h"
#include "relay/adapters/cooperative_adapters.h"
#include "relay/scheduler/edf_scheduler.h"

#include <cstddef>
#include <span>

namespace {

struct InstalledBackend
{
  [[nodiscard]] bool begin() noexcept { return true; }
  [[nodiscard]] fe::relay::BackendAdvance advance(std::size_t) noexcept
  {
    return {fe::relay::BackendStep::kComplete, 1uz};
  }
  void cancel() noexcept {}
  [[nodiscard]] std::size_t state_bytes() const noexcept { return 0uz; }
  [[nodiscard]] bool save_state(std::span<std::byte> state) const noexcept { return state.empty(); }
  [[nodiscard]] bool load_state(std::span<const std::byte> state) noexcept { return state.empty(); }
};

static_assert(fe::relay::CooperativeBackend<InstalledBackend>);

} // namespace

int main()
{
  fe_engine_free(nullptr);
  const fe::relay::EdfScheduler scheduler{1uz};
  InstalledBackend backend{};
  fe::relay::JobDescriptor descriptor{};
  descriptor.model_digest[0] = 1u;
  descriptor.state_schema = 1u;
  descriptor.total_work_units = 1u;
  auto job = fe::relay::make_iterative_job(backend, descriptor);
  if (!job || !job->start())
    return 1;
  const auto complete = job->advance(1uz);
  return scheduler.capacity() == 1uz && complete &&
                 complete->state == fe::relay::JobState::kComplete
             ? 0
             : 1;
}
