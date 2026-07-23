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

// Parse an F32/BF16 .safetensors file, copying each tensor into the arena as F32
[[nodiscard]] bool load_safetensors(std::string_view path, Arena& arena, std::span<TensorView> out,
                                    std::size_t& tensors_loaded) noexcept;

[[nodiscard]] std::size_t safetensors_f32_bytes(std::string_view path) noexcept;

// shape of a named tensor from the header
// for pre-load arena sizing
[[nodiscard]] std::array<std::size_t, 4> safetensors_tensor_shape(std::string_view path,
                                                                  std::string_view name) noexcept;

} // namespace fe
