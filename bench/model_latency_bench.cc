#include "api/engine.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <string_view>
#include <vector>

namespace {

std::atomic<std::size_t> g_allocations{0uz};

constexpr std::size_t kSequence = 4uz;

double percentile(const std::vector<double>& sorted, double quantile)
{
  const auto index = static_cast<std::size_t>(quantile * static_cast<double>(sorted.size() - 1uz));
  return sorted[index];
}

bool parse_size(std::string_view text, std::size_t& value) noexcept
{
  if (text.empty())
    return false;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc{} && end == text.data() + text.size();
}

int check_lifecycle(std::string_view label, const char* path, std::size_t cycles)
{
  if (cycles == 0uz)
    return 0;

  std::size_t expected_setup_allocations{0uz};
  bool have_expected_setup{false};
  const std::int32_t tokens[kSequence]{1, 2, 3, 4};
  for (std::size_t cycle{0uz}; cycle < cycles; ++cycle) {
    const std::size_t setup_before = g_allocations.load(std::memory_order_relaxed);
    std::unique_ptr<fe_engine, decltype(&fe_engine_free)> engine{fe_engine_load(path),
                                                                 fe_engine_free};
    const std::size_t setup_allocations =
        g_allocations.load(std::memory_order_relaxed) - setup_before;
    if (!engine) {
      std::fprintf(stderr, "%.*s lifecycle: %s\n", static_cast<int>(label.size()), label.data(),
                   fe_engine_last_error());
      return 1;
    }

    std::size_t d_model{0uz};
    std::size_t layers{0uz};
    fe_engine_dims(engine.get(), &d_model, &layers);
    std::vector<float> output(kSequence * d_model);
    const std::size_t hot_before = g_allocations.load(std::memory_order_relaxed);
    if (fe_engine_run(engine.get(), tokens, kSequence, output.data()) != 0) {
      std::fprintf(stderr, "%.*s lifecycle: %s\n", static_cast<int>(label.size()), label.data(),
                   fe_engine_last_error());
      return 1;
    }
    const std::size_t hot_allocations = g_allocations.load(std::memory_order_relaxed) - hot_before;
    if (hot_allocations != 0uz) {
      std::fprintf(stderr, "%.*s lifecycle: hot path allocated %zu times\n",
                   static_cast<int>(label.size()), label.data(), hot_allocations);
      return 1;
    }
    if (!have_expected_setup) {
      expected_setup_allocations = setup_allocations;
      have_expected_setup = true;
    } else if (setup_allocations != expected_setup_allocations) {
      std::fprintf(stderr, "%.*s lifecycle: setup allocations changed from %zu to %zu\n",
                   static_cast<int>(label.size()), label.data(), expected_setup_allocations,
                   setup_allocations);
      return 1;
    }
  }
  std::fprintf(stderr, "%.*s lifecycle: %zu cycles, %zu setup allocations per cycle, 0 hot\n",
               static_cast<int>(label.size()), label.data(), cycles, expected_setup_allocations);
  return 0;
}

int measure(std::string_view label, const char* path, std::size_t iterations,
            std::size_t lifecycle_cycles)
{
  using clock = std::chrono::steady_clock;
  if (check_lifecycle(label, path, lifecycle_cycles) != 0)
    return 1;

  const std::size_t setup_allocations_before = g_allocations.load(std::memory_order_relaxed);
  const auto load_start = clock::now();
  std::unique_ptr<fe_engine, decltype(&fe_engine_free)> engine{fe_engine_load(path),
                                                               fe_engine_free};
  const auto load_end = clock::now();
  const std::size_t setup_allocations =
      g_allocations.load(std::memory_order_relaxed) - setup_allocations_before;
  if (!engine) {
    std::fprintf(stderr, "%.*s: %s\n", static_cast<int>(label.size()), label.data(),
                 fe_engine_last_error());
    return 1;
  }

  std::size_t d_model{0uz};
  std::size_t layers{0uz};
  fe_engine_dims(engine.get(), &d_model, &layers);
  const std::int32_t tokens[kSequence]{1, 2, 3, 4};
  std::vector<float> output(kSequence * d_model);
  if (d_model == 0uz) {
    std::fprintf(stderr, "%.*s: benchmark requires a backbone checkpoint\n",
                 static_cast<int>(label.size()), label.data());
    return 1;
  }

  const auto cold_start = clock::now();
  if (fe_engine_run(engine.get(), tokens, kSequence, output.data()) != 0) {
    std::fprintf(stderr, "%.*s: %s\n", static_cast<int>(label.size()), label.data(),
                 fe_engine_last_error());
    return 1;
  }
  const auto cold_end = clock::now();
  for (unsigned i{0u}; i < 5u; ++i)
    static_cast<void>(fe_engine_run(engine.get(), tokens, kSequence, output.data()));

  std::vector<double> latency_ms;
  latency_ms.reserve(iterations);
  const std::size_t hot_allocations_before = g_allocations.load(std::memory_order_relaxed);
  float sink{0.0F};
  for (std::size_t i{0uz}; i < iterations; ++i) {
    const auto start = clock::now();
    static_cast<void>(fe_engine_run(engine.get(), tokens, kSequence, output.data()));
    const auto end = clock::now();
    latency_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    sink += output[i % output.size()];
  }
  const std::size_t hot_allocations =
      g_allocations.load(std::memory_order_relaxed) - hot_allocations_before;
  std::sort(latency_ms.begin(), latency_ms.end());
  double sum{0.0};
  for (const double value : latency_ms)
    sum += value;

  std::printf("%-8.*s | %7u | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f | %12zu | %10zu\n",
              static_cast<int>(label.size()), label.data(), fe_engine_thread_count(engine.get()),
              std::chrono::duration<double, std::milli>(load_end - load_start).count(),
              std::chrono::duration<double, std::milli>(cold_end - cold_start).count(),
              sum / static_cast<double>(latency_ms.size()), percentile(latency_ms, 0.50),
              percentile(latency_ms, 0.99), percentile(latency_ms, 0.999), setup_allocations,
              hot_allocations);
  return sink == 12345.678F ? 1 : 0;
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
  if (argc < 2 || argc > 6) {
    std::fputs("usage: flowedge_model_latency_bench F32_MODEL [BF16_MODEL] [ITERATIONS] "
               "[--lifecycle-cycles N]\n",
               stderr);
    return 2;
  }
  std::size_t next_argument{2uz};
  std::string_view bf16_model{};
  std::size_t iterations{100uz};
  if (next_argument < static_cast<std::size_t>(argc) &&
      std::string_view{argv[next_argument]} != "--lifecycle-cycles") {
    const std::string_view argument{argv[next_argument]};
    if (parse_size(argument, iterations))
      ++next_argument;
    else {
      bf16_model = argument;
      ++next_argument;
      if (next_argument < static_cast<std::size_t>(argc) &&
          std::string_view{argv[next_argument]} != "--lifecycle-cycles") {
        if (!parse_size(argv[next_argument], iterations)) {
          std::fputs("iterations must be a positive integer\n", stderr);
          return 2;
        }
        ++next_argument;
      }
    }
  }
  std::size_t lifecycle_cycles{0uz};
  if (next_argument < static_cast<std::size_t>(argc)) {
    if (next_argument + 2uz != static_cast<std::size_t>(argc) ||
        std::string_view{argv[next_argument]} != "--lifecycle-cycles") {
      std::fputs("expected --lifecycle-cycles N\n", stderr);
      return 2;
    }
    if (!parse_size(argv[next_argument + 1uz], lifecycle_cycles)) {
      std::fputs("lifecycle cycles must be a non-negative integer\n", stderr);
      return 2;
    }
  }
  if (iterations == 0uz) {
    std::fputs("iterations must be positive\n", stderr);
    return 2;
  }
  std::puts("Model    | Workers | Load(ms) | Cold(ms) | Mean(ms) |  p50(ms) |  p99(ms) | p999(ms) "
            "| Setup allocs | Hot allocs");
  std::puts("--------------------------------------------------------------------------------------"
            "-----------------------------");
  int result = measure("FP32", argv[1], iterations, lifecycle_cycles);
  if (!bf16_model.empty())
    result |= measure("BF16", bf16_model.data(), iterations, lifecycle_cycles);
  return result;
}
