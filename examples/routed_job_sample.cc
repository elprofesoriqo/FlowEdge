#include "relay/adapters/routed_adapters.h"
#include "relay/jobs/state_codec.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>

namespace {

using namespace fe::relay;

struct RefinementBackend
{
  [[nodiscard]] bool prepare(std::span<const std::byte> request) noexcept
  {
    StateReader reader{request};
    const auto initial = reader.read<std::uint64_t>();
    if (!initial || reader.remaining() != 0uz)
      return false;
    seed = *initial;
    return true;
  }

  [[nodiscard]] bool begin() noexcept
  {
    completed = 0u;
    value = seed;
    encode_result();
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
    encode_result();
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
    encode_result();
    return true;
  }

  [[nodiscard]] std::span<const std::byte> result() const noexcept { return output; }

  void encode_result() noexcept
  {
    StateWriter writer{output};
    static_cast<void>(writer.write(value));
  }

  static constexpr std::uint64_t kWorkUnits = 4u;
  std::uint64_t seed{};
  std::uint64_t completed{};
  std::uint64_t value{};
  std::array<std::byte, sizeof(std::uint64_t)> output{};
};

[[nodiscard]] JobDescriptor descriptor() noexcept
{
  JobDescriptor result{};
  result.kind = JobKind::kIterative;
  for (std::size_t index{}; index < result.model_digest.size(); ++index)
    result.model_digest[index] = static_cast<std::uint8_t>(index + 1uz);
  result.state_schema = 0x726f7574652d7631u; // "route-v1"
  result.session_id = 42u;
  result.generation = 7u;
  result.total_work_units = RefinementBackend::kWorkUnits;
  return result;
}

} // namespace

int main()
{
  std::array<std::byte, sizeof(std::uint64_t)> input{};
  StateWriter input_writer{input};
  if (!input_writer.write(std::uint64_t{5u}))
    return 1;

  auto request = std::make_unique<JobRequestMessage>();
  if (make_job_request(*request, 1u, descriptor(), 100u, input) != ProtocolResult::kSuccess)
    return 1;

  RefinementBackend backend{};
  std::array<JobAdapterRegistration, 1> registrations{};
  JobAdapterRegistry registry{registrations};
  if (!registry.add(make_routed_adapter(backend, job_route(descriptor()), input.size(),
                                        sizeof(std::uint64_t))))
    return 1;
  registry.freeze();

  auto routed = registry.bind(*request);
  if (!routed || !routed->start() || !routed->advance(2uz) || !routed->advance(2uz))
    return 1;

  auto result = std::make_unique<JobResultMessage>();
  if (routed->write_result(*result, 200u) != ProtocolResult::kSuccess ||
      validate(*result) != ProtocolResult::kSuccess)
    return 1;
  StateReader result_reader{result->payload_values()};
  const auto value = result_reader.read<std::uint64_t>();
  if (!value)
    return 1;

  std::cout << "route=iterative sequence=" << result->envelope.sequence
            << " outcome=" << to_string(job_result_code(*result))
            << " completed=" << result->metadata.progress.completed_work_units
            << " result=" << *value << '\n';
  return 0;
}
