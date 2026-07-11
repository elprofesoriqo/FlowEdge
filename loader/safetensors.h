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

struct SafetensorsLoader
{
  [[nodiscard]] bool load(std::string_view path, Arena& arena, std::span<TensorView> out,
                          std::size_t& tensors_loaded) const noexcept;
};

} // namespace fe
