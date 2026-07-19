#pragma once

#include "../arena/arena.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace fe {

struct TensorView
{
  float* data{};
  std::array<std::size_t, 4> shape{};
  std::array<char, 64> name{}; // null-terminated; safetensors keys are short
  std::uint8_t ndim{};

  [[nodiscard]] std::string_view name_view() const noexcept { return {name.data()}; }
};

// Parse an F32 .safetensors file, copying each tensor into the arena (64B-aligned,
// pre-faulted). Returns false on I/O error, malformed header, or if out is too small.
[[nodiscard]] bool load_safetensors(std::string_view path, Arena& arena, std::span<TensorView> out,
                                    std::size_t& tensors_loaded) noexcept;

} // namespace fe
