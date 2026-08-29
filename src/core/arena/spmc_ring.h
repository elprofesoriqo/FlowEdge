#pragma once

#include "arena/cpu.h"

#include <atomic>
#include <bit>
#include <cassert>
#include <cstddef>
#include <span>
#include <type_traits>

namespace fe {

template<typename T>
  requires std::is_trivially_copyable_v<T>
class SpmcRing
{
public:
  SpmcRing(std::span<T> slots, std::span<std::size_t> sequence) noexcept
      : slots_{slots}, sequence_{sequence}, mask_{slots.size() - 1uz},
        ok_{std::has_single_bit(slots.size()) && sequence.size() >= slots.size()}
  {
    assert(ok_);
    for (std::size_t i{0uz}; i < slots_.size(); ++i)
      sequence_[i] = i;
  }

  [[nodiscard]] bool valid() const noexcept { return ok_; }

  [[nodiscard]] bool try_push(const T& v) noexcept
  {
    if (!ok_) [[unlikely]]
      return false;

    const std::size_t pos = head_.load(std::memory_order_relaxed);
    std::size_t& seq = sequence_[pos & mask_];
    if (std::atomic_ref<std::size_t>{seq}.load(std::memory_order_acquire) != pos) [[unlikely]]
      return false;

    slots_[pos & mask_] = v;
    std::atomic_ref<std::size_t>{seq}.store(pos + 1uz, std::memory_order_release);
    head_.store(pos + 1uz, std::memory_order_relaxed);
    return true;
  }

  [[nodiscard]] bool try_pop(T& out) noexcept
  {
    if (!ok_) [[unlikely]]
      return false;

    std::size_t pos = tail_.load(std::memory_order_relaxed);
    for (;;) {
      std::size_t& seq = sequence_[pos & mask_];
      if (std::atomic_ref<std::size_t>{seq}.load(std::memory_order_acquire) != pos + 1uz)
        return false;

      if (tail_.compare_exchange_weak(pos, pos + 1uz, std::memory_order_acq_rel,
                                      std::memory_order_relaxed)) {
        out = slots_[pos & mask_];
        std::atomic_ref<std::size_t>{seq}.store(pos + slots_.size(), std::memory_order_release);
        return true;
      }
    }
  }

  [[nodiscard]] std::size_t capacity() const noexcept { return slots_.size(); }

private:
  std::span<T> slots_;
  std::span<std::size_t> sequence_;
  std::size_t mask_;
  bool ok_;

  alignas(kCacheLine) std::atomic<std::size_t> head_{0uz};
  alignas(kCacheLine) std::atomic<std::size_t> tail_{0uz};
};

} // namespace fe
