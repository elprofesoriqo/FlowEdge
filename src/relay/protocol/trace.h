#pragma once

#include "relay/protocol/job_events.h"
#include "relay/protocol/messages.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace fe::relay {

namespace detail {
struct TraceFileCloser
{
  void operator()(std::FILE* file) const noexcept;
};
using TraceFileHandle = std::unique_ptr<std::FILE, TraceFileCloser>;
} // namespace detail

struct TraceRecord
{
  MessageKind kind{};
  std::uint64_t sequence{};
  std::vector<std::byte> message{};
};

class TraceWriter
{
public:
  [[nodiscard]] static std::expected<TraceWriter, std::string> open(const char* path) noexcept;
  ~TraceWriter() = default;
  TraceWriter(const TraceWriter&) = delete;
  TraceWriter& operator=(const TraceWriter&) = delete;
  TraceWriter(TraceWriter&& other) noexcept = default;
  TraceWriter& operator=(TraceWriter&& other) noexcept = default;

  [[nodiscard]] std::expected<void, std::string> append(
      std::span<const std::byte> message) noexcept;

  [[nodiscard]] std::expected<void, std::string> append(const ConditionMessage& message) noexcept
  {
    return append(wire_bytes(message));
  }
  [[nodiscard]] std::expected<void, std::string> append(const ActionMessage& message) noexcept
  {
    return append(wire_bytes(message));
  }
  [[nodiscard]] std::expected<void, std::string> append(const JobRequestMessage& message) noexcept
  {
    return append(wire_bytes(message));
  }
  [[nodiscard]] std::expected<void, std::string> append(const JobResultMessage& message) noexcept
  {
    return append(wire_bytes(message));
  }
  [[nodiscard]] std::expected<void, std::string> append(const JobEventMessage& message) noexcept
  {
    return append(wire_bytes(message));
  }

  template<typename Message>
    requires std::is_trivially_copyable_v<Message>
  [[nodiscard]] std::expected<void, std::string> append(const Message& message) noexcept
  {
    return append(std::as_bytes(std::span{&message, 1uz}));
  }

  void flush() noexcept;

private:
  explicit TraceWriter(std::FILE* file) noexcept : file_{file} {}
  detail::TraceFileHandle file_{};
  static constexpr std::size_t kEncodedBytes =
      std::max({sizeof(ConditionMessage), sizeof(JobRequestMessage), sizeof(JobResultMessage),
                sizeof(JobEventMessage)});
  std::array<std::byte, kEncodedBytes> encoded_{};
};

class TraceReader
{
public:
  [[nodiscard]] static std::expected<TraceReader, std::string> open(const char* path) noexcept;
  ~TraceReader() = default;
  TraceReader(const TraceReader&) = delete;
  TraceReader& operator=(const TraceReader&) = delete;
  TraceReader(TraceReader&& other) noexcept = default;
  TraceReader& operator=(TraceReader&& other) noexcept = default;

  [[nodiscard]] std::expected<std::optional<TraceRecord>, std::string> next() noexcept;

private:
  explicit TraceReader(std::FILE* file, std::uint32_t version) noexcept
      : file_{file}, version_{version}
  {
  }
  detail::TraceFileHandle file_{};
  std::uint32_t version_{};
};

} // namespace fe::relay
