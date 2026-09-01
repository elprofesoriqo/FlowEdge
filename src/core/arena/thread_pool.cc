#include "arena/thread_pool.h"

#include "arena/cpu.h"

#include <algorithm>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace fe {
namespace {

void pin_thread(unsigned idx) noexcept
{
  const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
  const unsigned cpu = idx % hw;
#if defined(__linux__)
  cpu_set_t cs;
  CPU_ZERO(&cs);
  CPU_SET(cpu, &cs);
  pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
#elif defined(_WIN32)
  SetThreadAffinityMask(GetCurrentThread(), 1ULL << (cpu % 64u));
#else
  (void)cpu;
#endif
}

} // namespace

ThreadPool::ThreadPool(std::span<Task> ring, std::span<std::size_t> sequence,
                       std::span<std::jthread> workers, unsigned num_threads) noexcept
    : ring_{ring, sequence},
      workers_{workers.subspan(0uz, std::min(num_threads, static_cast<unsigned>(workers.size())))},
      nthreads_{static_cast<unsigned>(workers_.size())}
{
  if (!ring_.valid()) {
    nthreads_ = 0u;
    workers_ = {};
    return;
  }
  for (unsigned i{0u}; i < nthreads_; ++i)
    workers_[i] = std::jthread{&ThreadPool::worker_loop, this, i};
}

ThreadPool::~ThreadPool() noexcept
{
  stop_.store(true, std::memory_order_release);
  work_epoch_.fetch_add(1uz, std::memory_order_release);
  work_epoch_.notify_all();
  for (auto& w : workers_)
    if (w.joinable())
      w.join();
}

bool ThreadPool::enqueue(const Task& t) noexcept
{
  pending_.fetch_add(1uz, std::memory_order_acq_rel); // before publish so wait() cannot miss it
  if (ring_.try_push(t)) {
    work_epoch_.fetch_add(1uz, std::memory_order_release);
    work_epoch_.notify_one();
    return true;
  }
  pending_.fetch_sub(1uz, std::memory_order_acq_rel);
  return false;
}

void ThreadPool::wait() noexcept
{
  while (pending_.load(std::memory_order_acquire) != 0uz)
    cpu_pause();
}

FE_STACK_ALIGN void ThreadPool::worker_loop(unsigned idx) noexcept
{
  constexpr unsigned k_idle_spin = 4096u;
  pin_thread(idx);
  unsigned idle_spins{0u};
  for (;;) {
    Task t{};
    if (ring_.try_pop(t)) {
      idle_spins = 0u;
      t.fn(t.ctx, t.arg0, t.arg1);
      pending_.fetch_sub(1uz, std::memory_order_acq_rel);
      continue;
    }
    if (stop_.load(std::memory_order_acquire))
      break;
    if (idle_spins++ < k_idle_spin) {
      cpu_pause();
      continue;
    }

    // Load the epoch before checking the queue again. A producer racing this
    // transition either publishes work that we observe or changes the epoch,
    // causing atomic::wait to return immediately instead of losing a wakeup.
    const std::size_t observed = work_epoch_.load(std::memory_order_acquire);
    if (ring_.try_pop(t)) {
      idle_spins = 0u;
      t.fn(t.ctx, t.arg0, t.arg1);
      pending_.fetch_sub(1uz, std::memory_order_acq_rel);
      continue;
    }
    if (stop_.load(std::memory_order_acquire))
      break;
    work_epoch_.wait(observed, std::memory_order_acquire);
    idle_spins = 0u;
  }
}

} // namespace fe
