#pragma once

#include <bit>
#include <cassert>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <type_traits>

namespace fe {

inline constexpr std::size_t kSimdAlign = 64uz;

class Arena
{
public:
  constexpr explicit Arena(std::span<std::byte> slab) noexcept
      : cursor_{slab.data()}, end_{slab.data() + slab.size()}
  {
  }

  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;
  Arena(Arena&&) = delete;
  Arena& operator=(Arena&&) = delete;
  ~Arena() = default;

  template<std::size_t Align = alignof(std::max_align_t)>
  [[nodiscard]] void* alloc(std::size_t n) noexcept
  {
    static_assert(std::has_single_bit(Align));
    void* p = cursor_;
    auto space = static_cast<std::size_t>(end_ - cursor_);
    if (std::align(Align, n, p, space) == nullptr) [[unlikely]]
      return nullptr;
    cursor_ = static_cast<std::byte*>(p) + n;
    return p;
  }

  template<typename T, std::size_t Align = alignof(T)>
    requires std::is_trivially_copyable_v<T>
  [[nodiscard]] T* alloc_array(std::size_t count) noexcept
  {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) [[unlikely]]
      return nullptr;
    return static_cast<T*>(alloc<Align>(count * sizeof(T)));
  }

  template<typename T, std::size_t Align = alignof(T)>
    requires std::is_trivially_copyable_v<T>
  [[nodiscard]] std::span<T> alloc_span(std::size_t count) noexcept
  {
    T* const data = alloc_array<T, Align>(count);
    return data != nullptr ? std::span<T>{data, count} : std::span<T>{};
  }

  [[nodiscard]] std::byte* mark() const noexcept { return cursor_; }
  void reset_to(std::byte* mark) noexcept { cursor_ = mark; }

  [[nodiscard]] Arena sub_arena(std::size_t bytes) noexcept
  {
    void* p = alloc<kSimdAlign>(bytes);
    if (!p) [[unlikely]]
      return Arena{std::span<std::byte>{}};
    return Arena{std::span<std::byte>{static_cast<std::byte*>(p), bytes}};
  }

private:
  std::byte* cursor_;
  std::byte* const end_;
};

} // namespace fe
