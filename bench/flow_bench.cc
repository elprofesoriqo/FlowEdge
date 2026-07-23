#include "arena/arena.h"
#include "loader/safetensors.h"
#include "model/flow/flow.h"

#include <array>
#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace {

constexpr std::size_t kAction = 32uz;
constexpr std::size_t kCond = 768uz; // Mamba-130m d_model
constexpr std::size_t kHidden = 256uz;
constexpr std::size_t kTime = 128uz;
constexpr std::size_t kLayers = 4uz;
constexpr std::size_t kSteps = 10uz;

void fill(std::vector<float>& v)
{
  for (std::size_t i{0uz}; i < v.size(); ++i)
    v[i] = 0.02F * static_cast<float>(static_cast<int>(i % 17uz) - 8);
}

fe::TensorView view(const char* name, float* data, std::size_t rows, std::size_t cols)
{
  fe::TensorView v{};
  v.data = data;
  v.shape[0] = rows;
  v.shape[1] = cols;
  v.ndim = 2;
  std::size_t j{0uz};
  for (const char* p = name; (*p != '\0') && (j + 1uz < v.name.size()); ++p)
    v.name[j++] = *p;
  return v;
}

void run(benchmark::State& state, fe::FlowHead::Method method)
{
  std::vector<float> in(kHidden * kAction), tp(kHidden * kTime), cp(kHidden * kCond),
      op(kAction * kHidden);
  std::vector<std::vector<float>> layers(kLayers, std::vector<float>(kHidden * kHidden));
  fill(in);
  fill(tp);
  fill(cp);
  fill(op);
  for (auto& l : layers)
    fill(l);

  std::vector<fe::TensorView> views;
  views.push_back(view("flow.in_proj.weight", in.data(), kHidden, kAction));
  views.push_back(view("flow.time_proj.weight", tp.data(), kHidden, kTime));
  views.push_back(view("flow.cond_proj.weight", cp.data(), kHidden, kCond));
  views.push_back(view("flow.out_proj.weight", op.data(), kAction, kHidden));
  for (std::size_t l{0uz}; l < kLayers; ++l) {
    std::array<char, 32> nm{};
    std::snprintf(nm.data(), nm.size(), "flow.layers.%zu.weight", l);
    views.push_back(view(nm.data(), layers[l].data(), kHidden, kHidden));
  }

  std::vector<std::byte> slab(1uz << 20);
  fe::Arena arena{std::span<std::byte>{slab}};
  fe::FlowHead head{views, arena};
  if (!head.valid()) {
    state.SkipWithError("flow head invalid");
    return;
  }

  std::vector<float> cond(kCond, 0.1F), x0(kAction, 0.1F), out(kAction);
  for (auto _ : state) {
    head.sample(cond, x0, kSteps, method, out);
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()));
}

void BM_flow_euler(benchmark::State& state)
{
  run(state, fe::FlowHead::kEuler);
}
void BM_flow_heun(benchmark::State& state)
{
  run(state, fe::FlowHead::kHeun);
}

} // namespace

BENCHMARK(BM_flow_euler)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_flow_heun)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
