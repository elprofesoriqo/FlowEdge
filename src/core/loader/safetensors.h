#pragma once

#include "../arena/arena.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace fe {

struct TensorView
{
  enum class Dtype : std::uint8_t
  {
    F32,
    BF16,
    INT8
  };

  const void* data{};
  Dtype dtype{};
  std::array<std::size_t, 4> shape{};
  std::array<char, 64> name{}; // null-terminated; safetensors keys are short
  std::size_t bytes{};
  std::uint8_t ndim{};

  [[nodiscard]] std::string_view name_view() const noexcept { return {name.data()}; }

  [[nodiscard]] bool is_f32() const noexcept { return dtype == Dtype::F32; }
  [[nodiscard]] bool is_bf16() const noexcept { return dtype == Dtype::BF16; }

  [[nodiscard]] const float* as_f32() const noexcept
  {
    return is_f32() ? static_cast<const float*>(data) : nullptr;
  }
  [[nodiscard]] const uint16_t* as_bf16() const noexcept
  {
    return is_bf16() ? static_cast<const uint16_t*>(data) : nullptr;
  }
  [[nodiscard]] const uint8_t* as_i8() const noexcept
  {
    return (dtype == Dtype::INT8) ? static_cast<const uint8_t*>(data) : nullptr;
  }
};

struct WeightView
{
  const void* data{};
  TensorView::Dtype dtype{};
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

// One immutable, reference-counted checkpoint store. Engine instances retain a
// shared reference while keeping all scratch, sampler, and decode state private.
class ModelWeights
{
public:
  [[nodiscard]] static std::expected<std::shared_ptr<const ModelWeights>, const char*> open(
      std::string_view path) noexcept;

  ModelWeights(const ModelWeights&) = delete;
  ModelWeights& operator=(const ModelWeights&) = delete;
  ModelWeights(ModelWeights&&) = delete;
  ModelWeights& operator=(ModelWeights&&) = delete;
  ~ModelWeights() = default;

  [[nodiscard]] std::span<const TensorView> tensors() const noexcept
  {
    return {views_.data(), tensor_count_};
  }
  [[nodiscard]] std::size_t weight_bytes() const noexcept { return weight_bytes_; }

private:
  static constexpr std::size_t kMaxTensors = 1024uz;

  explicit ModelWeights(std::size_t weight_bytes);

  std::vector<std::byte> storage_{};
  Arena arena_;
  std::array<TensorView, kMaxTensors> views_{};
  std::size_t tensor_count_{};
  std::size_t weight_bytes_{};
};

} // namespace fe
