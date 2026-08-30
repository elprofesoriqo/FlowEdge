#include "arena/thread_pool.h"
#include "kernels/kernels.h"

#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace {

constexpr std::size_t kSeq = 4uz;
constexpr std::size_t kDModel = 768uz;
constexpr std::size_t kDInner = 1536uz;

std::vector<float> filled(std::size_t n, float base = 0.03F)
{
  std::vector<float> v(n);
  for (std::size_t i{0uz}; i < n; ++i)
    v[i] = base * static_cast<float>(static_cast<int>(i % 17uz) - 8);
  return v;
}

using benchmark::Counter;

void run_threaded_matmul(benchmark::State& state, std::size_t rows, std::size_t in_dim,
                         std::size_t out_dim)
{
  const unsigned threads = static_cast<unsigned>(state.range(0));

  std::vector<std::byte> ring_buf(64 * sizeof(fe::Task) + 64);
  void* ptr = ring_buf.data();
  std::size_t space = ring_buf.size();
  fe::Task* raw_ring = static_cast<fe::Task*>(std::align(64, 64 * sizeof(fe::Task), ptr, space));
  std::span<fe::Task> ring(raw_ring, 64);

  std::vector<std::size_t> seq(64);
  std::vector<std::jthread> workers(threads);
  fe::ThreadPool pool(ring, seq, workers, threads);

  const std::vector<float> in = filled(rows * in_dim);
  const std::vector<float> w = filled(out_dim * in_dim);
  std::vector<float> out(rows * out_dim);

  for (auto _ : state) {
    fe::matmul(in, w, out, rows, in_dim, out_dim, &pool);
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }

  const double flops = 2.0 * static_cast<double>(rows * in_dim * out_dim);
  state.counters["FLOP/s"] = Counter(flops, Counter::kIsIterationInvariantRate);
  state.counters["weight-B/s"] = Counter(static_cast<double>(out_dim * in_dim * sizeof(float)),
                                         Counter::kIsIterationInvariantRate);
}

void BM_matmul_in_proj_threaded(benchmark::State& state)
{
  run_threaded_matmul(state, kSeq, kDModel, 2uz * kDInner);
}

void BM_matmul_out_proj_threaded(benchmark::State& state)
{
  run_threaded_matmul(state, kSeq, kDInner, kDModel);
}

} // namespace

BENCHMARK(BM_matmul_in_proj_threaded)
    ->Unit(benchmark::kMicrosecond)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8);

BENCHMARK(BM_matmul_out_proj_threaded)
    ->Unit(benchmark::kMicrosecond)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8);

BENCHMARK_MAIN();
