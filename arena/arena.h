#pragma once

#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>

namespace fe {

inline constexpr std::size_t kSimdAlign = 64uz;

class Arena
{
public:
  constexpr explicit Arena(std::span<std::byte> slab) noexcept
      : begin_{slab.data()}, cursor_{slab.data()}, end_{slab.data() + slab.size()}
  {
  }

  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  [[nodiscard]] void* alloc(std::size_t n, std::size_t align = alignof(std::max_align_t)) noexcept
  {
    assert(std::has_single_bit(align));
    auto* p = reinterpret_cast<std::byte*>((reinterpret_cast<uintptr_t>(cursor_) + align - 1u) &
                                           ~(align - 1u));
    if (p > end_ || static_cast<std::size_t>(end_ - p) < n) [[unlikely]] // overflow-safe bound
      return nullptr;
    cursor_ = p + n;
    return p;
  }

  template<typename T>
    requires std::is_trivially_copyable_v<T>
  [[nodiscard]] T* alloc_array(std::size_t count, std::size_t align = alignof(T)) noexcept
  {
    return static_cast<T*>(alloc(count * sizeof(T), align));
  }

  void reset_to(std::byte* mark) noexcept { cursor_ = mark; }

  [[nodiscard]] std::byte* mark() const noexcept { return cursor_; }
  [[nodiscard]] std::size_t used() const noexcept
  {
    return static_cast<std::size_t>(cursor_ - begin_);
  }
  [[nodiscard]] std::size_t capacity() const noexcept
  {
    return static_cast<std::size_t>(end_ - begin_);
  }

private:
  std::byte* const begin_;
  std::byte* cursor_;
  std::byte* const end_;
};

} // namespace fe
