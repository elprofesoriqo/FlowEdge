#pragma once

#include "arena/cpu.h"
#include "arena/spmc_ring.h"

#include <atomic>
#include <concepts>
#include <cstddef>
#include <new>
#include <span>
#include <thread>
#include <type_traits>

namespace fe {

struct alignas(kCacheLine) Task
{
  void (*fn)(void* ctx, std::size_t lo, std::size_t hi) noexcept {};
  void* ctx{};
  std::size_t arg0{};
  std::size_t arg1{};
};
static_assert(sizeof(Task) <= kCacheLine);

// workers spin on an SPMC ring
class ThreadPool
{
public:
  // ring.size() must be 2^n. sequence.size() >= ring.size().
  ThreadPool(std::span<Task> ring, std::span<std::size_t> sequence, std::span<std::jthread> workers,
             unsigned num_threads) noexcept;
  ~ThreadPool() noexcept;

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  // false if the ring is full
  [[nodiscard]] bool enqueue(const Task& t) noexcept;

  void wait() noexcept;

  [[nodiscard]] unsigned nthreads() const noexcept { return nthreads_; }

private:
#if defined(_WIN32) && defined(__GNUC__)
  __attribute__((force_align_arg_pointer))
#endif
  void
  worker_loop(unsigned idx) noexcept;

  SpmcRing<Task> ring_;
  std::span<std::jthread> workers_;
  unsigned nthreads_{0u};

  alignas(kCacheLine) std::atomic<std::size_t> pending_{0uz};
  alignas(kCacheLine) std::atomic<bool> stop_{false};
};

template<typename Fn>
#if defined(_WIN32) && defined(__GNUC__)
__attribute__((force_align_arg_pointer))
#endif
static void
trampoline(void* ctx, std::size_t lo, std::size_t hi) noexcept
{
  (*static_cast<std::remove_reference_t<Fn>*>(ctx))(lo, hi);
}

template<typename Fn>
  requires std::invocable<Fn, std::size_t, std::size_t>
void parallel_for(ThreadPool& pool, std::size_t total, Fn&& fn) noexcept
{
  const std::size_t n = pool.nthreads();
  if (n == 0uz) {
    fn(0uz, total);
    return;
  }
  const std::size_t sz = (total + n - 1uz) / n;
  for (std::size_t i{0uz}; i < n && (i * sz) < total; ++i) {
    const std::size_t lo = i * sz;
    const std::size_t hi = (lo + sz < total) ? (lo + sz) : total;
    while (!pool.enqueue({trampoline<Fn>, static_cast<void*>(&fn), lo, hi}))
      cpu_pause();
  }
  pool.wait();
}

} // namespace fe
