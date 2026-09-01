#include "kernels/kernels.h"

#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

// Mamba-130M shapes.
constexpr std::size_t kSeq = 4uz;
constexpr std::size_t kDModel = 768uz;
constexpr std::size_t kDInner = 1536uz;
constexpr std::size_t kDState = 16uz;
constexpr std::size_t kDConv = 4uz;
constexpr std::size_t kDtRank = 48uz;
constexpr std::size_t kProjW = kDtRank + (2uz * kDState); // x_proj output width

std::vector<float> filled(std::size_t n, float base = 0.03F)
{
  std::vector<float> v(n);
  for (std::size_t i{0uz}; i < n; ++i)
    v[i] = base * static_cast<float>(static_cast<int>(i % 17uz) - 8); // small, mixed-sign
  return v;
}

using benchmark::Counter;

// out[r,o] = Σ_i in[r,i]·w[o,i]
// FLOP/s and streamed-weight bytes/s.
void run_matmul(benchmark::State& state, std::size_t rows, std::size_t in_dim, std::size_t out_dim)
{
  const std::vector<float> in = filled(rows * in_dim);
  const std::vector<float> w = filled(out_dim * in_dim);
  std::vector<float> out(rows * out_dim);
  for (auto _ : state) {
    fe::matmul(in, w, out, rows, in_dim, out_dim);
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  const double flops = 2.0 * static_cast<double>(rows * in_dim * out_dim);
  state.counters["FLOP/s"] = Counter(flops, Counter::kIsIterationInvariantRate);
  state.counters["weight-B/s"] = Counter(static_cast<double>(out_dim * in_dim * sizeof(float)),
                                         Counter::kIsIterationInvariantRate);
}

void BM_matmul_in_proj(benchmark::State& state)
{
  run_matmul(state, kSeq, kDModel, 2uz * kDInner);
}
void BM_matmul_out_proj(benchmark::State& state)
{
  run_matmul(state, kSeq, kDInner, kDModel);
}
void BM_matmul_x_proj(benchmark::State& state)
{
  run_matmul(state, kSeq, kDInner, kProjW);
}
void BM_matmul_dt_proj(benchmark::State& state)
{
  run_matmul(state, kSeq, kDtRank, kDInner);
}

void BM_silu(benchmark::State& state)
{
  std::vector<float> x = filled(kSeq * kDInner);
  for (auto _ : state) {
    fe::silu(x);
    benchmark::DoNotOptimize(x.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * x.size()));
}

void BM_softplus(benchmark::State& state)
{
  std::vector<float> x = filled(kSeq * kDInner);
  for (auto _ : state) {
    fe::softplus(x);
    benchmark::DoNotOptimize(x.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * x.size()));
}

void BM_gate_silu(benchmark::State& state)
{
  const std::vector<float> a = filled(kSeq * kDInner);
  const std::vector<float> g = filled(kSeq * kDInner, 0.02F);
  std::vector<float> out(kSeq * kDInner);
  for (auto _ : state) {
    fe::gate_silu(a, g, out);
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * out.size()));
}

void BM_rmsnorm(benchmark::State& state)
{
  const std::vector<float> in = filled(kSeq * kDModel);
  const std::vector<float> w = filled(kDModel, 1.0F);
  std::vector<float> out(kSeq * kDModel);
  for (auto _ : state) {
    fe::rmsnorm(in, w, out, kSeq, kDModel);
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * out.size()));
}

void BM_conv1d_causal(benchmark::State& state)
{
  const std::vector<float> x = filled(kDInner * kSeq);
  const std::vector<float> w = filled(kDInner * kDConv);
  const std::vector<float> b = filled(kDInner);
  std::vector<float> y(kDInner * kSeq);
  for (auto _ : state) {
    fe::conv1d_causal(x, w, b, y, kDInner, kSeq, kDConv);
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * y.size()));
}

void BM_discretize_and_scan(benchmark::State& state)
{
  const std::vector<float> dt = filled(kSeq * kDInner, 0.01F);
  const std::vector<float> a_neg = filled(kDInner * kDState, -1.0F);
  const std::vector<float> b = filled(kSeq * kDState);
  const std::vector<float> u = filled(kSeq * kDInner);
  const std::vector<float> c = filled(kSeq * kDState);
  const std::vector<float> d = filled(kDInner);
  std::vector<float> h(kDState * kDInner);
  std::vector<float> y(kSeq * kDInner);
  for (auto _ : state) {
    fe::discretize_and_scan(dt, a_neg, b, u, c, d, h, y, kSeq, kDInner, kDState, true);
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * y.size()));
}

} // namespace

BENCHMARK(BM_matmul_in_proj)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_matmul_out_proj)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_matmul_x_proj)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_matmul_dt_proj)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_silu)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_softplus)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_gate_silu)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_rmsnorm)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_conv1d_causal)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_discretize_and_scan)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
