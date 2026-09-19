#pragma once

#include <cstdlib>

namespace fe {

// True iff NAME=1. Engine load reads this at process start; callers must not
// mutate the environment while another thread is loading an engine.
[[nodiscard]] inline bool environment_flag(const char* name) noexcept
{
  const char* const value = std::getenv(name); // NOLINT(concurrency-mt-unsafe)
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

} // namespace fe
