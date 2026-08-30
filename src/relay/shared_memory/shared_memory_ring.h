#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <type_traits>

namespace fe::relay {

struct RingConfig
{
  std::uint32_t capacity{16u};
  std::uint32_t slot_bytes{};
};

enum class RingResult : std::uint8_t
{
  kSuccess,
  kEmpty,
  kFull,
  kTooLarge,
  kDestinationTooSmall,
  kCorrupt,
};

class SharedMemoryRing
{
public:
  [[nodiscard]] static std::expected<SharedMemoryRing, std::string> create(
      std::string name, RingConfig config, bool unlink_on_destroy = true) noexcept;
  [[nodiscard]] static std::expected<SharedMemoryRing, std::string> open(std::string name) noexcept;

  SharedMemoryRing() = default;
  ~SharedMemoryRing();
  SharedMemoryRing(const SharedMemoryRing&) = delete;
  SharedMemoryRing& operator=(const SharedMemoryRing&) = delete;
  SharedMemoryRing(SharedMemoryRing&& other) noexcept;
  SharedMemoryRing& operator=(SharedMemoryRing&& other) noexcept;

  [[nodiscard]] bool valid() const noexcept { return mapping_ != nullptr; }
  [[nodiscard]] RingConfig config() const noexcept;
  [[nodiscard]] std::uint64_t size() const noexcept;
  [[nodiscard]] RingResult try_push(std::span<const std::byte> payload) noexcept;
  [[nodiscard]] RingResult try_pop(std::span<std::byte> destination,
                                   std::size_t& bytes_read) noexcept;

  template<typename Message>
    requires std::is_trivially_copyable_v<Message>
  [[nodiscard]] RingResult try_push(const Message& message) noexcept
  {
    return try_push(std::as_bytes(std::span{&message, 1uz}));
  }

private:
  void release() noexcept;

  void* mapping_{};
  std::size_t mapping_bytes_{};
  std::string name_{};
  bool unlink_on_destroy_{false};
#ifdef _WIN32
  void* handle_{};
#else
  int descriptor_{-1};
#endif
};

} // namespace fe::relay
