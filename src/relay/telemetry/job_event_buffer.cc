#include "relay/telemetry/job_event_buffer.h"

#include <exception>

namespace fe::relay {

std::expected<JobEventBuffer, std::string> JobEventBuffer::create(std::size_t capacity) noexcept
{
  if (capacity == 0uz)
    return std::unexpected("Generic Relay event capacity must be non-zero");
  try {
    JobEventBuffer buffer{};
    buffer.storage_.resize(capacity);
    return buffer;
  } catch (const std::exception& error) {
    return std::unexpected("Failed to allocate generic Relay event buffer: " +
                           std::string{error.what()});
  } catch (...) {
    return std::unexpected("Failed to allocate generic Relay event buffer");
  }
}

JobEventBufferResult JobEventBuffer::try_push(const JobEventMessage& event) noexcept
{
  if (validate(event) != ProtocolResult::kSuccess)
    return JobEventBufferResult::kInvalid;
  if (size_ == storage_.size()) {
    ++dropped_;
    return JobEventBufferResult::kFull;
  }
  storage_[write_] = event;
  write_ = (write_ + 1uz) % storage_.size();
  ++size_;
  return JobEventBufferResult::kSuccess;
}

const JobEventMessage* JobEventBuffer::front() const noexcept
{
  return empty() ? nullptr : &storage_[read_];
}

void JobEventBuffer::pop() noexcept
{
  if (empty())
    return;
  read_ = (read_ + 1uz) % storage_.size();
  --size_;
}

} // namespace fe::relay
