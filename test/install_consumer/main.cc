#include "api/engine.h"
#include "relay/adapters/routed_adapters.h"
#include "relay/scheduler/edf_scheduler.h"

#include <array>
#include <cstddef>
#include <span>

namespace {

struct InstalledBackend
{
  [[nodiscard]] bool prepare(std::span<const std::byte> request) noexcept
  {
    return request.empty();
  }
  [[nodiscard]] bool begin() noexcept { return true; }
  [[nodiscard]] fe::relay::BackendAdvance advance(std::size_t) noexcept
  {
    return {fe::relay::BackendStep::kComplete, 1uz};
  }
  void cancel() noexcept {}
  [[nodiscard]] std::size_t state_bytes() const noexcept { return 0uz; }
  [[nodiscard]] bool save_state(std::span<std::byte> state) const noexcept { return state.empty(); }
  [[nodiscard]] bool load_state(std::span<const std::byte> state) noexcept { return state.empty(); }
  [[nodiscard]] std::span<const std::byte> result() const noexcept { return {}; }
};

static_assert(fe::relay::CooperativeBackend<InstalledBackend>);
static_assert(fe::relay::RoutedBackend<InstalledBackend>);

} // namespace

int main()
{
  fe_engine_free(nullptr);
  const fe::relay::EdfScheduler scheduler{1uz};
  InstalledBackend backend{};
  fe::relay::JobDescriptor descriptor{};
  descriptor.kind = fe::relay::JobKind::kIterative;
  descriptor.model_digest[0] = 1u;
  descriptor.state_schema = 1u;
  descriptor.total_work_units = 1u;
  auto job = fe::relay::make_iterative_job(backend, descriptor);
  if (!job || !job->start())
    return 1;
  const auto complete = job->advance(1uz);
  std::array<fe::relay::JobAdapterRegistration, 1> entries{};
  fe::relay::JobAdapterRegistry registry{entries};
  if (!registry.add(
          fe::relay::make_routed_adapter(backend, fe::relay::job_route(descriptor), 0uz, 0uz)))
    return 1;
  registry.freeze();
  fe::relay::JobRequestMessage request{};
  if (fe::relay::make_job_request(request, 1u, descriptor, 0u, {}) !=
      fe::relay::ProtocolResult::kSuccess)
    return 1;
  auto routed = registry.bind(request);
  if (!routed || !routed->start())
    return 1;
  const auto routed_complete = routed->advance(1uz);
  return scheduler.capacity() == 1uz && complete && routed_complete &&
                 complete->state == fe::relay::JobState::kComplete &&
                 routed_complete->state == fe::relay::JobState::kComplete
             ? 0
             : 1;
}
