#pragma once

#include "loader/safetensors.h"
#include "protocol/contracts.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace fe {

using ModelDigest = std::array<std::uint8_t, FE_MODEL_DIGEST_BYTES>;

struct ModelIdentity
{
  ModelDigest digest{};
  std::uint32_t architecture{FE_ARCH_UNKNOWN};
  std::uint32_t precision{FE_PRECISION_UNKNOWN};
  std::uint64_t d_model{};
  std::uint64_t n_layers{};
  std::uint64_t d_inner{};
  std::uint64_t d_state{};
  std::uint64_t d_conv{};
  std::uint64_t action_dim{};
  std::uint64_t condition_dim{};

  [[nodiscard]] friend bool operator==(const ModelIdentity&, const ModelIdentity&) = default;
};

// Stable content fingerprints used for compatibility, replay, and corruption
// detection. They are not authentication primitives.
[[nodiscard]] ModelDigest fingerprint_bytes(std::span<const std::byte> bytes) noexcept;
[[nodiscard]] ModelDigest fingerprint_tensors(std::span<const TensorView> tensors) noexcept;
[[nodiscard]] std::uint32_t checkpoint_precision(std::span<const TensorView> tensors) noexcept;

} // namespace fe
