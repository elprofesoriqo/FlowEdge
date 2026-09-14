#include "api/engine.h"

#include <atomic>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>
#include <optional>
#include <string_view>
#include <vector>

namespace {

std::atomic<std::size_t> g_allocations{0uz};

bool parse_size(std::string_view text, std::size_t& value) noexcept
{
  if (text.empty())
    return false;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc{} && end == text.data() + text.size();
}

bool parse_token(std::string_view text, std::int32_t& value) noexcept
{
  if (text.empty())
    return false;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc{} && end == text.data() + text.size();
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
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " MODEL TOKEN [TOKEN ...] [--cycles N]\n";
    return 2;
  }
  std::size_t cycles{3uz};
  std::vector<std::int32_t> tokens;
  tokens.reserve(static_cast<std::size_t>(argc - 2));
  for (int index{2}; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "--cycles") {
      if (++index == argc || !parse_size(argv[index], cycles) || cycles == 0uz) {
        std::cerr << "error: --cycles requires a positive integer\n";
        return 2;
      }
      continue;
    }
    std::int32_t token{0};
    if (!parse_token(argument, token)) {
      std::cerr << "error: token must be a signed 32-bit integer: " << argument << '\n';
      return 2;
    }
    tokens.push_back(token);
  }
  if (tokens.empty()) {
    std::cerr << "error: supply at least one token\n";
    return 2;
  }

  std::optional<std::size_t> setup_allocations;
  for (std::size_t cycle{0uz}; cycle < cycles; ++cycle) {
    const std::size_t setup_before = g_allocations.load(std::memory_order_relaxed);
    std::unique_ptr<fe_engine, decltype(&fe_engine_free)> engine{fe_engine_load(argv[1]),
                                                                 fe_engine_free};
    const std::size_t setup = g_allocations.load(std::memory_order_relaxed) - setup_before;
    if (!engine) {
      std::cerr << "error: " << fe_engine_last_error() << '\n';
      return 1;
    }
    if (setup_allocations && *setup_allocations != setup) {
      std::cerr << "error: setup allocation count changed between lifecycle cycles\n";
      return 1;
    }
    setup_allocations = setup;

    std::size_t d_model{0uz};
    std::size_t layers{0uz};
    fe_engine_dims(engine.get(), &d_model, &layers);
    if (d_model == 0uz) {
      std::cerr << "error: lifecycle check requires a backbone checkpoint\n";
      return 1;
    }
    std::vector<float> output(tokens.size() * d_model);
    const std::size_t hot_before = g_allocations.load(std::memory_order_relaxed);
    if (fe_engine_run(engine.get(), tokens.data(), tokens.size(), output.data()) != 0) {
      std::cerr << "error: " << fe_engine_last_error() << '\n';
      return 1;
    }
    const std::size_t hot = g_allocations.load(std::memory_order_relaxed) - hot_before;
    if (hot != 0uz) {
      std::cerr << "error: hot path allocated " << hot << " times\n";
      return 1;
    }
  }
  std::cout << "{\"schema_version\":1,\"cycles\":" << cycles
            << ",\"setup_allocations\":" << *setup_allocations << ",\"hot_path_allocations\":0}\n";
  return 0;
}
