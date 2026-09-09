#include "api/engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <new>
#include <string_view>
#include <vector>

namespace {

std::atomic<std::size_t> g_allocations{0uz};

struct EngineDeleter
{
  void operator()(fe_engine* engine) const noexcept { fe_engine_free(engine); }
};

double percentile(const std::vector<double>& sorted, double percentile_value)
{
  const auto index = static_cast<std::size_t>(
      std::ceil(percentile_value * static_cast<double>(sorted.size())) - 1.0);
  return sorted[std::min(index, sorted.size() - 1uz)];
}

template<typename Function> std::vector<double> measure(std::size_t iterations, Function&& function)
{
  std::vector<double> samples;
  samples.reserve(iterations);
  for (std::size_t i{0uz}; i < iterations; ++i) {
    const auto begin = std::chrono::steady_clock::now();
    if (function() != 0)
      return {};
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
  }
  std::ranges::sort(samples);
  return samples;
}

void print_result(std::string_view name, const std::vector<double>& samples)
{
  std::cout << name << "_us p50=" << percentile(samples, 0.50)
            << " p95=" << percentile(samples, 0.95) << " p99=" << percentile(samples, 0.99) << '\n';
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
  if (argc < 2) {
    std::cerr
        << "usage: flowedge_diffusion_latency_bench <converted.safetensors> [steps] [iters]\n";
    return 1;
  }
  const std::size_t steps = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 10uz;
  const std::size_t iterations = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 100uz;
  if (steps == 0uz || iterations == 0uz)
    return 1;
  const std::unique_ptr<fe_engine, EngineDeleter> engine{fe_engine_load(argv[1])};
  if (!engine) {
    std::cerr << fe_engine_last_error() << '\n';
    return 2;
  }
  const std::size_t horizon = fe_engine_action_horizon(engine.get());
  const std::size_t action_dim = fe_engine_action_dim(engine.get());
  const std::size_t condition_dim = fe_engine_condition_dim(engine.get());
  if (horizon <= 1uz || action_dim == 0uz || condition_dim == 0uz)
    return 3;
  std::vector<float> condition(condition_dim, 0.0F);
  std::vector<float> noise(horizon * action_dim);
  std::vector<float> output(noise.size());
  for (std::size_t i{0uz}; i < noise.size(); ++i)
    noise[i] = std::sin(static_cast<float>(i + 1uz) * 0.31F);

  for (int warmup{0}; warmup < 2; ++warmup)
    if (fe_engine_sample_diffusion(engine.get(), condition.data(), noise.data(), steps,
                                   FE_DIFFUSION_DDIM, 0u, output.data()) != 0)
      return 4;
  const std::vector<double> denoiser = measure(iterations, [&] {
    return fe_engine_diffusion_denoise(engine.get(), condition.data(), noise.data(), 50.0F,
                                       output.data());
  });
  const std::vector<double> sampler = measure(iterations, [&] {
    return fe_engine_sample_diffusion(engine.get(), condition.data(), noise.data(), steps,
                                      FE_DIFFUSION_DDIM, 0u, output.data());
  });
  if (denoiser.empty() || sampler.empty()) {
    std::cerr << fe_engine_last_error() << '\n';
    return 5;
  }
  const std::size_t allocations_before = g_allocations.load(std::memory_order_relaxed);
  for (int check{0}; check < 3; ++check) {
    if (fe_engine_diffusion_denoise(engine.get(), condition.data(), noise.data(), 50.0F,
                                    output.data()) != 0 ||
        fe_engine_sample_diffusion(engine.get(), condition.data(), noise.data(), steps,
                                   FE_DIFFUSION_DDIM, 0u, output.data()) != 0)
      return 5;
  }
  const std::size_t hot_path_allocations =
      g_allocations.load(std::memory_order_relaxed) - allocations_before;

#if defined(__clang__)
  constexpr std::string_view compiler = "clang " __clang_version__;
#elif defined(__GNUC__)
  constexpr std::string_view compiler = "gcc " __VERSION__;
#elif defined(_MSC_VER)
  constexpr std::string_view compiler = "msvc";
#else
  constexpr std::string_view compiler = "unknown";
#endif
#if defined(NDEBUG)
  constexpr std::string_view build = "release";
#else
  constexpr std::string_view build = "debug";
#endif
  const char* const cpu = std::getenv("FLOWEDGE_BENCH_CPU");
  std::cout << std::fixed << std::setprecision(3) << "cpu=" << (cpu ? cpu : "unspecified")
            << " compiler=" << compiler << " build=" << build
            << " threads=" << fe_engine_thread_count(engine.get()) << " steps=" << steps
            << " iterations=" << iterations << " hot_path_allocations=" << hot_path_allocations
            << '\n';
  print_result("denoiser", denoiser);
  print_result("ddim", sampler);
  return hot_path_allocations == 0uz ? 0 : 6;
}
