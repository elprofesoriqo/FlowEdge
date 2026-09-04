#include "relay/adapters/routed_adapters.h"
#include "relay/jobs/state_codec.h"
#include "relay/worker/job_service.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <thread>

namespace {

using namespace fe::relay;

struct RefinementBackend
{
  [[nodiscard]] bool prepare(std::span<const std::byte> request) noexcept
  {
    StateReader reader{request};
    const auto decoded = reader.read<std::uint64_t>();
    if (!decoded || reader.remaining() != 0uz)
      return false;
    seed = *decoded;
    return true;
  }

  [[nodiscard]] bool begin() noexcept
  {
    completed = 0u;
    value = seed;
    update_result();
    return true;
  }

  [[nodiscard]] BackendAdvance advance(std::size_t budget) noexcept
  {
    const std::size_t count =
        static_cast<std::size_t>(std::min<std::uint64_t>(budget, kWorkUnits - completed));
    for (std::size_t index{}; index < count; ++index) {
      value = (value * 3u) + completed + 1u;
      ++completed;
    }
    update_result();
    return {
        .step = completed == kWorkUnits ? BackendStep::kComplete : BackendStep::kInProgress,
        .completed_work_units = count,
    };
  }

  void cancel() noexcept {}
  [[nodiscard]] std::size_t state_bytes() const noexcept { return 16uz; }

  [[nodiscard]] bool save_state(std::span<std::byte> destination) const noexcept
  {
    StateWriter writer{destination};
    return writer.write(completed) && writer.write(value) && writer.remaining() == 0uz;
  }

  [[nodiscard]] bool load_state(std::span<const std::byte> source) noexcept
  {
    StateReader reader{source};
    const auto next_completed = reader.read<std::uint64_t>();
    const auto next_value = reader.read<std::uint64_t>();
    if (!next_completed || !next_value || *next_completed > kWorkUnits || reader.remaining() != 0uz)
      return false;
    completed = *next_completed;
    value = *next_value;
    update_result();
    return true;
  }

  [[nodiscard]] std::span<const std::byte> result() const noexcept { return encoded_result; }

  void update_result() noexcept
  {
    StateWriter writer{encoded_result};
    static_cast<void>(writer.write(value));
  }

  static constexpr std::uint64_t kWorkUnits = 4u;
  std::uint64_t seed{};
  std::uint64_t completed{};
  std::uint64_t value{};
  std::array<std::byte, sizeof(std::uint64_t)> encoded_result{};
};

[[nodiscard]] JobDescriptor descriptor() noexcept
{
  JobDescriptor value{};
  value.kind = JobKind::kIterative;
  for (std::size_t index{}; index < value.model_digest.size(); ++index)
    value.model_digest[index] = static_cast<std::uint8_t>(index + 1uz);
  value.state_schema = 0x726f7574652d7631u; // "route-v1"
  value.total_work_units = RefinementBackend::kWorkUnits;
  return value;
}

} // namespace

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::cerr << "usage: flowedge_job_service_fixture REQUEST_SHM RESULT_SHM\n";
    return 2;
  }

  RefinementBackend backend{};
  std::array<JobAdapterRegistration, 1> entries{};
  std::array<JobAdapterRegistry, 1> lanes{JobAdapterRegistry{entries}};
  if (!lanes[0].add(make_routed_adapter(backend, job_route(descriptor()), sizeof(std::uint64_t),
                                        sizeof(std::uint64_t))))
    return 1;
  lanes[0].freeze();
  auto created_pool = JobWorkerPool::create(lanes, 8uz, {}, 8uz, 1uz);
  if (!created_pool) {
    std::cerr << created_pool.error() << '\n';
    return 1;
  }
  JobWorkerPool pool = std::move(*created_pool);
  auto created_service = JobService::create(argv[1], argv[2], pool, 8u);
  if (!created_service) {
    std::cerr << created_service.error() << '\n';
    return 1;
  }
  JobService service = std::move(*created_service);

  while (!service.stopped()) {
    const JobServiceResult result = service.poll();
    if (result == JobServiceResult::kCorruptInput || result == JobServiceResult::kTransportError) {
      std::cerr << "generic job service: " << to_string(result) << '\n';
      return 1;
    }
    if (result == JobServiceResult::kIdle || result == JobServiceResult::kBackpressured)
      std::this_thread::yield();
  }

  const JobTransportStats& stats = service.stats();
  std::cout << "generic job service stopped: accepted=" << stats.requests_accepted
            << " rejected=" << stats.requests_rejected << " published=" << stats.results_published
            << '\n';
  return 0;
}
