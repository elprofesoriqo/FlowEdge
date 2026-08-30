#include "protocol/model_identity.h"

#include <bit>
#include <cstring>
#include <string_view>

namespace fe {
namespace {

constexpr std::uint64_t kPrime1 = 11400714785074694791ULL;
constexpr std::uint64_t kPrime2 = 14029467366897019727ULL;
constexpr std::uint64_t kPrime3 = 1609587929392839161ULL;
constexpr std::uint64_t kPrime4 = 9650029242287828579ULL;
constexpr std::uint64_t kPrime5 = 2870177450012600261ULL;

[[nodiscard]] std::uint64_t read_u64(const std::byte* source) noexcept
{
  std::uint64_t value{0u};
  std::memcpy(&value, source, sizeof(value));
  if constexpr (std::endian::native == std::endian::big)
    value = std::byteswap(value);
  return value;
}

[[nodiscard]] std::uint32_t read_u32(const std::byte* source) noexcept
{
  std::uint32_t value{0u};
  std::memcpy(&value, source, sizeof(value));
  if constexpr (std::endian::native == std::endian::big)
    value = std::byteswap(value);
  return value;
}

[[nodiscard]] constexpr std::uint64_t round(std::uint64_t accumulator, std::uint64_t input) noexcept
{
  accumulator += input * kPrime2;
  accumulator = std::rotl(accumulator, 31);
  return accumulator * kPrime1;
}

[[nodiscard]] constexpr std::uint64_t avalanche(std::uint64_t value) noexcept
{
  value ^= value >> 33u;
  value *= kPrime2;
  value ^= value >> 29u;
  value *= kPrime3;
  return value ^ (value >> 32u);
}

[[nodiscard]] std::uint64_t hash64(std::span<const std::byte> bytes, std::uint64_t seed) noexcept
{
  if (bytes.empty())
    return avalanche(seed + kPrime5);
  const std::byte* cursor = bytes.data();
  const std::byte* const end = cursor + bytes.size();
  std::uint64_t hash{0u};
  if (bytes.size() >= 32uz) {
    std::uint64_t lane1 = seed + kPrime1 + kPrime2;
    std::uint64_t lane2 = seed + kPrime2;
    std::uint64_t lane3 = seed;
    std::uint64_t lane4 = seed - kPrime1;
    const std::byte* const block_end = end - 32uz;
    while (cursor <= block_end) {
      lane1 = round(lane1, read_u64(cursor));
      lane2 = round(lane2, read_u64(cursor + 8uz));
      lane3 = round(lane3, read_u64(cursor + 16uz));
      lane4 = round(lane4, read_u64(cursor + 24uz));
      cursor += 32uz;
    }
    hash = std::rotl(lane1, 1) + std::rotl(lane2, 7) + std::rotl(lane3, 12) + std::rotl(lane4, 18);
    for (const std::uint64_t lane : {lane1, lane2, lane3, lane4}) {
      hash ^= round(0u, lane);
      hash = (hash * kPrime1) + kPrime4;
    }
  } else {
    hash = seed + kPrime5;
  }

  hash += bytes.size();
  while (cursor + 8uz <= end) {
    hash ^= round(0u, read_u64(cursor));
    hash = (std::rotl(hash, 27) * kPrime1) + kPrime4;
    cursor += 8uz;
  }
  if (cursor + 4uz <= end) {
    hash ^= static_cast<std::uint64_t>(read_u32(cursor)) * kPrime1;
    hash = (std::rotl(hash, 23) * kPrime2) + kPrime3;
    cursor += 4uz;
  }
  while (cursor < end) {
    hash ^= std::to_integer<std::uint8_t>(*cursor) * kPrime5;
    hash = std::rotl(hash, 11) * kPrime1;
    ++cursor;
  }
  return avalanche(hash);
}

[[nodiscard]] constexpr std::uint64_t combine(std::uint64_t accumulator,
                                              std::uint64_t value) noexcept
{
  return (std::rotl(accumulator ^ round(0u, value), 27) * kPrime1) + kPrime4;
}

[[nodiscard]] ModelDigest make_digest(std::uint64_t first, std::uint64_t second) noexcept
{
  ModelDigest result{};
  for (std::size_t i{0uz}; i < 8uz; ++i) {
    result[i] = static_cast<std::uint8_t>(first >> (8uz * i));
    result[8uz + i] = static_cast<std::uint8_t>(second >> (8uz * i));
  }
  return result;
}

} // namespace

ModelDigest fingerprint_bytes(std::span<const std::byte> bytes) noexcept
{
  const std::uint64_t first = hash64(bytes, 0x243f6a8885a308d3ULL);
  const std::uint64_t second =
      avalanche(first ^ static_cast<std::uint64_t>(bytes.size()) ^ 0x13198a2e03707344ULL);
  return make_digest(first, second);
}

ModelDigest fingerprint_tensors(std::span<const TensorView> tensors) noexcept
{
  std::uint64_t first = combine(0x243f6a8885a308d3ULL, tensors.size());
  std::uint64_t second = combine(0x13198a2e03707344ULL, tensors.size());
  for (const TensorView& tensor : tensors) {
    const std::string_view name = tensor.name_view();
    const std::uint64_t name_hash =
        hash64(std::as_bytes(std::span{name.data(), name.size()}), kPrime3);
    first = combine(first, name_hash);
    second = combine(second, std::rotl(name_hash, 17));
    const std::uint64_t format =
        static_cast<std::uint64_t>(tensor.dtype) | (static_cast<std::uint64_t>(tensor.ndim) << 8u);
    first = combine(first, format);
    second = combine(second, format ^ tensor.bytes);
    for (std::size_t i{0uz}; i < tensor.ndim; ++i) {
      first = combine(first, tensor.shape[i]);
      second = combine(second, std::rotl(static_cast<std::uint64_t>(tensor.shape[i]), 23));
    }
    const std::uint64_t content =
        hash64({static_cast<const std::byte*>(tensor.data), tensor.bytes}, kPrime5);
    first = combine(first, content);
    second = combine(second, std::rotl(content, 31) ^ tensor.bytes);
  }
  return make_digest(avalanche(first), avalanche(second));
}

std::uint32_t checkpoint_precision(std::span<const TensorView> tensors) noexcept
{
  bool f32{false};
  bool bf16{false};
  for (const TensorView& tensor : tensors) {
    f32 |= tensor.dtype == TensorView::Dtype::F32;
    bf16 |= tensor.dtype == TensorView::Dtype::BF16;
  }
  if (f32 && bf16)
    return FE_PRECISION_MIXED;
  if (bf16)
    return FE_PRECISION_BF16;
  if (f32)
    return FE_PRECISION_F32;
  return FE_PRECISION_UNKNOWN;
}

} // namespace fe
