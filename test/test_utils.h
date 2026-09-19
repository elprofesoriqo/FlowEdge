#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace fe::test {

inline std::vector<float> seq(std::size_t size, float frequency, float phase)
{
  std::vector<float> values(size);
  for (std::size_t index{}; index < size; ++index)
    values[index] = std::sin(static_cast<float>(index) * frequency + phase);
  return values;
}

inline std::vector<std::uint16_t> to_bf16(const std::vector<float>& values)
{
  std::vector<std::uint16_t> result(values.size());
  for (std::size_t index{}; index < values.size(); ++index) {
    std::uint32_t bits{};
    std::memcpy(&bits, &values[index], sizeof(bits));
    result[index] = static_cast<std::uint16_t>(bits >> 16);
  }
  return result;
}

inline void set_process_env(const char* name, const char* value)
{
  static std::map<std::string, std::string> storage;
  std::string& slot = storage[name];
  slot = std::string(name) + "=" + (value != nullptr ? value : "");
#ifdef _WIN32
  _putenv(slot.c_str()); // NOLINT(concurrency-mt-unsafe)
#else
  if (value == nullptr || value[0] == '\0')
    unsetenv(name);         // NOLINT(concurrency-mt-unsafe)
  else
    setenv(name, value, 1); // NOLINT(concurrency-mt-unsafe)
#endif
}

class EnvOverride
{
public:
  EnvOverride(const char* name, const char* value) : name_(name)
  {
    const char* const previous = std::getenv(name); // NOLINT(concurrency-mt-unsafe)
    had_ = previous != nullptr;
    if (had_)
      previous_ = previous;
    set_process_env(name_.c_str(), value);
  }

  EnvOverride(const EnvOverride&) = delete;
  EnvOverride& operator=(const EnvOverride&) = delete;

  ~EnvOverride()
  {
    if (had_)
      set_process_env(name_.c_str(), previous_.c_str());
    else
      set_process_env(name_.c_str(), "");
  }

private:
  std::string name_;
  std::string previous_;
  bool had_{};
};

inline constexpr float k_tol{2e-3F};

} // namespace fe::test
