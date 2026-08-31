#include "relay/adapters/cooperative_adapters.h"
#include "relay/jobs/state_codec.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <new>
#include <span>
#include <string_view>

namespace {

using namespace fe::relay;

std::atomic<std::size_t> g_allocations{};

struct CounterBackend
{
  [[nodiscard]] bool begin() noexcept
  {
    completed = 0u;
    value = 0x9e3779b97f4a7c15ULL;
    return true;
  }

  [[nodiscard]] BackendAdvance advance(std::size_t budget) noexcept
  {
    const std::size_t count =
        static_cast<std::size_t>(std::min<std::uint64_t>(budget, kWorkUnits - completed));
    for (std::size_t index{}; index < count; ++index) {
      value = (value ^ (completed + 1u)) * 0xbf58476d1ce4e5b9ULL;
      ++completed;
    }
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
    return true;
  }

  static constexpr std::uint64_t kWorkUnits = 8u;
  std::uint64_t completed{};
  std::uint64_t value{};
};

[[nodiscard]] JobDescriptor descriptor() noexcept
{
  JobDescriptor result{};
  for (std::size_t index{}; index < result.model_digest.size(); ++index)
    result.model_digest[index] = static_cast<std::uint8_t>(index + 1uz);
  result.state_schema = 0x62656e63682d7631u;
  result.session_id = 1u;
  result.generation = 1u;
  result.total_work_units = CounterBackend::kWorkUnits;
  return result;
}

} // namespace

void* operator new(std::size_t bytes)
{
  g_allocations.fetch_add(1uz, std::memory_order_relaxed);
  if (void* memory = std::malloc(bytes == 0uz ? 1uz : bytes))
    return memory;
  throw std::bad_alloc{};
}

void operator delete(void* memory) noexcept
{
  std::free(memory);
}
void operator delete(void* memory, std::size_t) noexcept
{
  std::free(memory);
}
void* operator new[](std::size_t bytes)
{
  return ::operator new(bytes);
}
void operator delete[](void* memory) noexcept
{
  std::free(memory);
}
void operator delete[](void* memory, std::size_t) noexcept
{
  std::free(memory);
}

int main(int argc, char** argv)
{
  std::size_t iterations{100'000uz};
  if (argc > 1) {
    const std::string_view input{argv[1]};
    const auto [end, error] =
        std::from_chars(input.data(), input.data() + input.size(), iterations);
    if (error != std::errc{} || end != input.data() + input.size() || iterations == 0uz) {
      std::cerr << "Usage: flowedge_cooperative_job_bench [iterations]\n";
      return 2;
    }
  }

  std::array<std::byte, 256> capsule{};
  std::uint64_t checksum{};
  const std::size_t allocations_before = g_allocations.load(std::memory_order_relaxed);
  const auto start = std::chrono::steady_clock::now();
  for (std::size_t iteration{}; iteration < iterations; ++iteration) {
    CounterBackend source_backend{};
    auto source = make_iterative_job(source_backend, descriptor());
    if (!source || !source->start() || !source->advance(3uz))
      return 1;
    const auto bytes = source->export_capsule(capsule);
    if (!bytes)
      return 1;
    CounterBackend destination_backend{};
    auto destination = make_iterative_job(destination_backend, descriptor());
    if (!destination || !destination->restore_capsule(std::span{capsule}.first(*bytes)) ||
        !destination->advance(CounterBackend::kWorkUnits))
      return 1;
    checksum ^= destination_backend.value + iteration;
  }
  const auto elapsed = std::chrono::steady_clock::now() - start;
  const std::size_t allocations =
      g_allocations.load(std::memory_order_relaxed) - allocations_before;
  const double elapsed_ns =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
  const double ns_per_iteration = elapsed_ns / static_cast<double>(iterations);
  const double ns_per_work_unit =
      elapsed_ns / static_cast<double>(iterations * CounterBackend::kWorkUnits);
  std::cout << "iterations=" << iterations << " ns_per_migrated_job=" << ns_per_iteration
            << " ns_per_work_unit=" << ns_per_work_unit << " hot_path_allocations=" << allocations
            << " checksum=" << checksum << '\n';
  return allocations == 0uz ? 0 : 1;
}
