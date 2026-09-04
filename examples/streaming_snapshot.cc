#include "api/engine.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <ranges>
#include <span>
#include <vector>

namespace {

struct EngineDeleter
{
  void operator()(fe_engine* engine) const noexcept { fe_engine_free(engine); }
};

using Engine = std::unique_ptr<fe_engine, EngineDeleter>;

[[nodiscard]] bool step(fe_engine* engine, std::int32_t token, std::span<float> output)
{
  if (fe_engine_step(engine, token, output.data()) == 0)
    return true;
  std::cerr << "streaming step failed: " << fe_engine_last_error() << '\n';
  return false;
}

} // namespace

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cerr << "usage: streaming_snapshot <model.safetensors>\n";
    return 2;
  }

  Engine source{fe_engine_load_with_threads(argv[1], 0u)};
  Engine migrated{fe_engine_load_with_threads(argv[1], 0u)};
  if (!source || !migrated) {
    std::cerr << "load failed: " << fe_engine_last_error() << '\n';
    return 1;
  }

  std::size_t dimension{};
  fe_engine_dims(source.get(), &dimension, nullptr);
  const std::size_t snapshot_bytes = fe_engine_decode_state_bytes(source.get());
  if (dimension == 0uz || snapshot_bytes == 0uz) {
    std::cerr << "checkpoint has no streaming backbone state\n";
    return 1;
  }

  std::vector<float> source_output(dimension);
  std::vector<float> migrated_output(dimension);
  constexpr std::array<std::int32_t, 4uz> prefix{1, 2, 3, 4};
  for (const std::int32_t token : prefix)
    if (!step(source.get(), token, source_output))
      return 1;

  std::vector<std::byte> snapshot(snapshot_bytes);
  if (fe_engine_export_decode_state(source.get(), snapshot.data(), snapshot.size()) != 0 ||
      fe_engine_import_decode_state(migrated.get(), snapshot.data(), snapshot.size()) != 0) {
    std::cerr << "state migration failed: " << fe_engine_last_error() << '\n';
    return 1;
  }

  constexpr std::int32_t branch_token = 5;
  if (!step(source.get(), branch_token, source_output) ||
      !step(migrated.get(), branch_token, migrated_output))
    return 1;

  const bool exact = std::ranges::equal(source_output, migrated_output);
  std::cout << "snapshot_bytes=" << snapshot_bytes
            << " migrated_exact=" << (exact ? "true" : "false")
            << " output0=" << source_output.front() << '\n';
  return exact ? 0 : 1;
}
