#pragma once

#include "../arena/arena.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string_view>

namespace fe {

struct TensorView
{
  enum class Dtype : std::uint8_t
  {
    F32,
    BF16,
    INT8
  };

  void* data{};
  Dtype dtype{};
  std::array<std::size_t, 4> shape{};
  std::array<char, 64> name{}; // null-terminated; safetensors keys are short
  std::uint8_t ndim{};

  [[nodiscard]] std::string_view name_view() const noexcept { return {name.data()}; }

  // typed data access
  [[nodiscard]] const float* as_f32() const noexcept { return static_cast<const float*>(data); }
  [[nodiscard]] const uint16_t* as_bf16() const noexcept
  {
    return static_cast<const uint16_t*>(data);
  }
  [[nodiscard]] const uint8_t* as_i8() const noexcept { return static_cast<const uint8_t*>(data); }
};

// First tensor whose name matches, or nullptr.
[[nodiscard]] inline const TensorView* find_tensor(std::span<const TensorView> ts,
                                                   std::string_view name) noexcept
{
  for (const auto& t : ts)
    if (t.name_view() == name)
      return &t;
  return nullptr;
}

// Parse an F32/BF16 .safetensors file; weights stored as-is
[[nodiscard]] std::expected<void, const char*> load_safetensors(
    std::string_view path, Arena& arena, std::span<TensorView> out,
    std::size_t& tensors_loaded) noexcept;

// returns actual stored bytes
[[nodiscard]] std::size_t safetensors_weight_bytes(std::string_view path) noexcept;

// shape of a named tensor from the header, for pre-load arena sizing
[[nodiscard]] std::array<std::size_t, 4> safetensors_tensor_shape(std::string_view path,
                                                                  std::string_view name) noexcept;

} // namespace fe
