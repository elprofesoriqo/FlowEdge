#pragma once

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#endif

#include <cstddef>

namespace fe {

inline constexpr std::size_t kCacheLine = 64uz;

inline void cpu_pause() noexcept
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
  _mm_pause();
#elif defined(__aarch64__) || defined(__arm__)
  __asm__ volatile("yield" ::: "memory");
#endif
}

} // namespace fe
