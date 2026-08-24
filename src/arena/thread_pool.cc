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
    workers_[i] = std::jthread{[this, i] {
      worker_loop(i);
    }};
}

ThreadPool::~ThreadPool() noexcept
{
  stop_.store(true, std::memory_order_release);
  for (auto& w : workers_)
    if (w.joinable())
      w.join();
}

bool ThreadPool::enqueue(const Task& t) noexcept
{
  pending_.fetch_add(1uz, std::memory_order_acq_rel); // before publish so wait() cannot miss it
  if (ring_.try_push(t))
    return true;
  pending_.fetch_sub(1uz, std::memory_order_acq_rel);
  return false;
}

void ThreadPool::wait() noexcept
{
  while (pending_.load(std::memory_order_acquire) != 0uz)
    cpu_pause();
}

#if defined(_WIN32) && defined(__GNUC__)
__attribute__((force_align_arg_pointer))
#endif
void ThreadPool::worker_loop(unsigned idx) noexcept
{
  pin_thread(idx);
  for (;;) {
    Task t{};
    if (ring_.try_pop(t)) {
      t.fn(t.ctx, t.arg0, t.arg1);
      pending_.fetch_sub(1uz, std::memory_order_acq_rel);
      continue;
    }
    if (stop_.load(std::memory_order_acquire))
      break;
    cpu_pause();
  }
}

} // namespace fe
