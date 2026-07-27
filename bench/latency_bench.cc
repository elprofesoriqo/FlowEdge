#include "arena/arena.h"
#include "heads/flow/flow.h"
#include "loader/safetensors.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <span>
#include <string_view>
#include <vector>

namespace {

std::atomic<std::size_t> g_allocs{0uz};

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

double pct(const std::vector<double>& sorted, double q)
{
  const auto i = static_cast<std::size_t>(q * static_cast<double>(sorted.size() - 1uz));
  return sorted[i];
}

} // namespace

void* operator new(std::size_t n)
{
  g_allocs.fetch_add(1uz, std::memory_order_relaxed);
  if (void* p = std::malloc(n == 0uz ? 1uz : n))
    return p;
  throw std::bad_alloc{};
}
void operator delete(void* p) noexcept
{
  std::free(p);
}
void operator delete(void* p, std::size_t) noexcept
{
  std::free(p);
}
void* operator new[](std::size_t n)
{
  return ::operator new(n);
}
void operator delete[](void* p) noexcept
{
  std::free(p);
}
void operator delete[](void* p, std::size_t) noexcept
{
  std::free(p);
}

int main(int argc, char** argv)
{
  const std::string_view m_arg = (argc > 1) ? argv[1] : "";
  const auto method = (m_arg == "rk4")    ? fe::FlowHead::kRK4
                      : (m_arg == "heun") ? fe::FlowHead::kHeun
                                          : fe::FlowHead::kEuler;
  const char* method_name = (method == fe::FlowHead::kRK4)    ? "rk4"
                            : (method == fe::FlowHead::kHeun) ? "heun"
                                                              : "euler";

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
    std::puts("flow head invalid");
    return 1;
  }

  std::vector<float> cond(kCond, 0.1F), x0(kAction, 0.1F), out(kAction);
  constexpr std::size_t kWarm = 2000uz;
  constexpr std::size_t kIters = 100000uz;
  for (std::size_t i{0uz}; i < kWarm; ++i)
    head.sample(cond, x0, kSteps, method, out);

  std::vector<double> us;
  us.reserve(kIters); // pre-allocated: the timed loop must not allocate
  float sink = 0.0F;
  using clock = std::chrono::steady_clock;
  const std::size_t allocs_before = g_allocs.load(std::memory_order_relaxed);
  for (std::size_t i{0uz}; i < kIters; ++i) {
    const auto t0 = clock::now();
    head.sample(cond, x0, kSteps, method, out);
    const auto t1 = clock::now();
    us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    sink += out[0];
  }
  const std::size_t hot_allocs = g_allocs.load(std::memory_order_relaxed) - allocs_before;

  std::sort(us.begin(), us.end());
  double sum = 0.0;
  for (const double v : us)
    sum += v;
  const double mean = sum / static_cast<double>(us.size());
  const double p50 = pct(us, 0.50);
  const double p99 = pct(us, 0.99);
  const double p999 = pct(us, 0.999);

  std::printf("FlowEdge   | %9.2f | %9.2f | %9.2f | %9.2f | %9.2f | %9.2f | %zu\n", mean, p50, p99,
              p999, us.front(), us.back(), hot_allocs);
  return (sink == 12345.678F) ? 1 : 0;
}
