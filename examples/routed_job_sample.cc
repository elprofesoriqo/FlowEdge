#include "relay/adapters/routed_adapters.h"
#include "relay/jobs/state_codec.h"
#include "relay/protocol/trace.h"
#include "relay/telemetry/metrics.h"
#include "relay/worker/job_worker_pool.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <thread>

namespace {

using namespace fe::relay;

[[nodiscard]] std::uint64_t monotonic_ns() noexcept
{
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now().time_since_epoch())
                                        .count());
}

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

int main(int argc, char** argv)
{
  if (argc > 2) {
    std::cerr << "usage: routed_job_sample [trace-file]\n";
    return 2;
  }
  std::optional<TraceWriter> trace{};
  std::unique_ptr<JobResultMessage> trace_result{};
  if (argc == 2) {
    auto opened = TraceWriter::open(argv[1]);
    if (!opened) {
      std::cerr << opened.error() << '\n';
      return 1;
    }
    trace.emplace(std::move(*opened));
    trace_result = std::make_unique<JobResultMessage>();
  }

  std::array<std::byte, sizeof(std::uint64_t)> input{};
  StateWriter input_writer{input};
  if (!input_writer.write(std::uint64_t{5u}))
    return 1;

  auto request = std::make_unique<JobRequestMessage>();
  const std::uint64_t submitted_ns = monotonic_ns();
  if (make_job_request(*request, 1u, descriptor(), submitted_ns, input) != ProtocolResult::kSuccess)
    return 1;

  RefinementBackend backend{};
  std::array<JobAdapterRegistration, 1> registrations{};
  JobAdapterRegistry registry{registrations};
  if (!registry.add(make_routed_adapter(backend, job_route(descriptor()), input.size(),
                                        sizeof(std::uint64_t))))
    return 1;
  registry.freeze();

  std::array<JobAdapterRegistry, 1> lanes{registry};
  auto events = JobEventBuffer::create(8uz);
  if (!events)
    return 1;
  auto created = JobWorkerPool::create(lanes, 4uz, {}, 4uz, 2uz, WorkerPlacement::kNone, &*events);
  if (!created)
    return 1;
  JobWorkerPool pool = std::move(*created);
  if (trace && !trace->append(*request))
    return 1;
  if (pool.submit(*request, submitted_ns) != JobSubmitResult::kAccepted)
    return 1;

  const JobResultMessage* result{};
  while (result == nullptr) {
    result = pool.ready_result();
    if (result == nullptr)
      std::this_thread::yield();
  }
  if (validate(*result) != ProtocolResult::kSuccess)
    return 1;
  StateReader result_reader{result->payload_values()};
  const auto value = result_reader.read<std::uint64_t>();
  if (!value)
    return 1;
  if (trace_result)
    std::memcpy(trace_result.get(), result, wire_size(*result));

  std::cout << "route=iterative sequence=" << result->envelope.sequence
            << " outcome=" << to_string(job_result_code(*result))
            << " completed=" << result->metadata.progress.completed_work_units
            << " result=" << *value;
  pool.release_ready_result();
  JobMetrics metrics{};
  while (const JobEventMessage* event = events->front()) {
    metrics.record(*event);
    if (trace && !trace->append(*event))
      return 1;
    events->pop();
  }
  if (trace && !trace->append(*trace_result))
    return 1;
  if (trace)
    trace->flush();
  std::cout << " events=" << metrics.counters().events
            << " work_units=" << metrics.counters().completed_work_units;
  if (argc == 2)
    std::cout << " trace=" << argv[1];
  std::cout << '\n';
  return pool.release_session(descriptor().session_id) && metrics.counters().completed == 1u ? 0
                                                                                             : 1;
}
