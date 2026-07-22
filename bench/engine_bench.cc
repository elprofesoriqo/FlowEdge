#include "api/engine.h"

#include <array>
#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace {

void BM_engine_forward(benchmark::State& state)
{
  const char* path = std::getenv("FLOWEDGE_MODEL");
  fe_engine* engine = fe_engine_load((path != nullptr) ? path : "models/mamba.safetensors");
  if (engine == nullptr) {
    state.SkipWithError("no checkpoint");
    return;
  }
  std::size_t d_model{0uz};
  std::size_t n_layers{0uz};
  fe_engine_dims(engine, &d_model, &n_layers);

  const std::array<std::int32_t, 4> tokens{1, 2, 3, 4};
  std::vector<float> out(tokens.size() * d_model);
  for (auto _ : state) {
    const int rc = fe_engine_run(engine, tokens.data(), tokens.size(), out.data());
    benchmark::DoNotOptimize(out.data());
    if (rc != 0) {
      state.SkipWithError("forward failed");
      break;
    }
  }
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()));
  fe_engine_free(engine);
}

} // namespace

BENCHMARK(BM_engine_forward)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();