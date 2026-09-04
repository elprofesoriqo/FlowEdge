#pragma once

#include "relay/protocol/job_events.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace fe::relay {

enum class JobEventBufferResult : std::uint8_t
{
  kSuccess,
  kInvalid,
  kFull,
};

// Single-coordinator bounded event queue. Storage is allocated by create();
// push, front, and pop do not allocate. Drain it outside the compute path.
class JobEventBuffer
{
public:
  [[nodiscard]] static std::expected<JobEventBuffer, std::string> create(
      std::size_t capacity) noexcept;

  [[nodiscard]] JobEventBufferResult try_push(const JobEventMessage& event) noexcept;
  [[nodiscard]] const JobEventMessage* front() const noexcept;
  void pop() noexcept;

  [[nodiscard]] std::size_t capacity() const noexcept { return storage_.size(); }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0uz; }
  [[nodiscard]] std::uint64_t dropped() const noexcept { return dropped_; }

private:
  std::vector<JobEventMessage> storage_{};
  std::size_t read_{};
  std::size_t write_{};
  std::size_t size_{};
  std::uint64_t dropped_{};
};

} // namespace fe::relay
