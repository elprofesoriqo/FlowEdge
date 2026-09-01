#pragma once

#include "arena/cpu.h"
#include "arena/spmc_ring.h"

#include <algorithm>
#include <atomic>
#include <concepts>
#include <cstddef>
#include <memory>
#include <new>
#include <span>
#include <thread>
#include <type_traits>
#include <utility>

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
  FE_STACK_ALIGN void worker_loop(unsigned idx) noexcept;

  SpmcRing<Task> ring_;
  std::span<std::jthread> workers_;
  unsigned nthreads_{0u};

  alignas(kCacheLine) std::atomic<std::size_t> pending_{0uz};
  alignas(kCacheLine) std::atomic<std::size_t> work_epoch_{0uz};
  alignas(kCacheLine) std::atomic<bool> stop_{false};
};

template<typename Fn>
FE_STACK_ALIGN inline void trampoline(void* ctx, std::size_t lo, std::size_t hi) noexcept
{
  (*static_cast<std::remove_reference_t<Fn>*>(ctx))(lo, hi);
}

template<typename Fn>
  requires std::invocable<Fn, std::size_t, std::size_t>
void parallel_for(ThreadPool& pool, std::size_t total, unsigned task_count, Fn&& fn) noexcept
{
  const std::size_t n = std::min<std::size_t>({pool.nthreads(), task_count, total});
  if (n <= 1uz) {
    fn(0uz, total);
    return;
  }
  const std::size_t sz = (total + n - 1uz) / n;
  for (std::size_t i{0uz}; i < n && (i * sz) < total; ++i) {
    const std::size_t lo = i * sz;
    const std::size_t hi = (lo + sz < total) ? (lo + sz) : total;
    void* const context = const_cast<void*>(static_cast<const void*>(std::addressof(fn)));
    while (!pool.enqueue({trampoline<Fn>, context, lo, hi}))
      cpu_pause();
  }
  pool.wait();
}

template<typename Fn>
  requires std::invocable<Fn, std::size_t, std::size_t>
void parallel_for(ThreadPool& pool, std::size_t total, Fn&& fn) noexcept
{
  parallel_for(pool, total, pool.nthreads(), std::forward<Fn>(fn));
}

} // namespace fe
