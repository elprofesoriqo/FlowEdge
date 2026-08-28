#pragma once

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#endif

#include <cstddef>

namespace fe {

inline constexpr std::size_t kCacheLine = 64uz;

#if defined(_WIN32) && defined(__GNUC__)
#define FE_STACK_ALIGN __attribute__((force_align_arg_pointer))
#else
#define FE_STACK_ALIGN
#endif

inline void cpu_pause() noexcept
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
  _mm_pause();
#elif defined(__aarch64__) || defined(__arm__)
  __asm__ volatile("yield" ::: "memory");
#endif
}

} // namespace fe
