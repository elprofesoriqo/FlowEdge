#include "relay/protocol/trace.h"

#include "protocol/model_identity.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

namespace fe::relay {
namespace {

constexpr std::array<char, 8> kTraceMagic{'F', 'E', 'T', 'R', 'A', 'C', 'E', '1'};
constexpr std::uint32_t kTraceVersion = 1u;
constexpr std::uint32_t kRecordMagic = 0x31434552u; // "REC1"

struct FileHeader
{
  std::array<char, 8> magic{kTraceMagic};
  std::uint32_t version{kTraceVersion};
  std::uint32_t header_bytes{16u};
};
static_assert(sizeof(FileHeader) == 16);

struct RecordHeader
{
  std::uint32_t magic{kRecordMagic};
  std::uint16_t version{1u};
  MessageKind kind{};
  std::uint32_t message_bytes{};
  std::uint32_t reserved{};
  std::uint64_t sequence{};
  std::uint64_t checksum{};
};
static_assert(sizeof(RecordHeader) == 32);

[[nodiscard]] std::uint64_t checksum(std::span<const std::byte> bytes) noexcept
{
  const ModelDigest digest = fingerprint_bytes(bytes);
  std::uint64_t value{0u};
  for (std::size_t i{0uz}; i < sizeof(value); ++i)
    value |= static_cast<std::uint64_t>(digest[i]) << (8uz * i);
  return value;
}

[[nodiscard]] bool write_all(std::FILE* file, const void* data, std::size_t bytes) noexcept
{
  return std::fwrite(data, 1uz, bytes, file) == bytes;
}

[[nodiscard]] bool read_all(std::FILE* file, void* data, std::size_t bytes) noexcept
{
  return std::fread(data, 1uz, bytes, file) == bytes;
}

[[nodiscard]] constexpr bool valid_kind(MessageKind kind) noexcept
{
  return kind == MessageKind::kCondition || kind == MessageKind::kAction ||
         kind == MessageKind::kShutdown;
}

} // namespace

void detail::TraceFileCloser::operator()(std::FILE* file) const noexcept
{
  if (file != nullptr)
    std::fclose(file); // NOLINT(cppcoreguidelines-owning-memory): unique_ptr deleter.
}

std::expected<TraceWriter, std::string> TraceWriter::open(const char* path) noexcept
{
  if (path == nullptr)
    return std::unexpected("Trace path is null");
  detail::TraceFileHandle file{std::fopen(path, "wb")};
  if (!file)
    return std::unexpected("Failed to create Relay trace");
  const FileHeader header{};
  if (!write_all(file.get(), &header, sizeof(header))) {
    return std::unexpected("Failed to write Relay trace header");
  }
  return TraceWriter{file.release()};
}

std::expected<void, std::string> TraceWriter::append(std::span<const std::byte> message) noexcept
{
  if (file_ == nullptr || message.size() < sizeof(MessageEnvelope) ||
      message.size() > std::numeric_limits<std::uint32_t>::max())
    return std::unexpected("Relay trace message is invalid");
  MessageEnvelope envelope{};
  std::memcpy(&envelope, message.data(), sizeof(envelope));
  if (envelope.magic != kMessageMagic || envelope.version != kMessageVersion ||
      !valid_kind(envelope.kind) || envelope.struct_size != message.size())
    return std::unexpected("Relay trace message envelope is invalid");
  const RecordHeader record{.kind = envelope.kind,
                            .message_bytes = static_cast<std::uint32_t>(message.size()),
                            .sequence = envelope.sequence,
                            .checksum = checksum(message)};
  if (!write_all(file_.get(), &record, sizeof(record)) ||
      !write_all(file_.get(), message.data(), message.size()))
    return std::unexpected("Failed to append Relay trace record");
  return {};
}

void TraceWriter::flush() noexcept
{
  if (file_ != nullptr)
    std::fflush(file_.get());
}

std::expected<TraceReader, std::string> TraceReader::open(const char* path) noexcept
{
  if (path == nullptr)
    return std::unexpected("Trace path is null");
  detail::TraceFileHandle file{std::fopen(path, "rb")};
  if (!file)
    return std::unexpected("Failed to open Relay trace");
  FileHeader header{};
  if (!read_all(file.get(), &header, sizeof(header)) || header.magic != kTraceMagic ||
      header.version != kTraceVersion || header.header_bytes != sizeof(header)) {
    return std::unexpected("Relay trace header is incompatible or corrupt");
  }
  return TraceReader{file.release()};
}

std::expected<std::optional<TraceRecord>, std::string> TraceReader::next() noexcept
{
  if (file_ == nullptr)
    return std::unexpected("Relay trace reader is closed");
  RecordHeader header{};
  const std::size_t bytes = std::fread(&header, 1uz, sizeof(header), file_.get());
  if (bytes == 0uz && std::feof(file_.get()) != 0)
    return std::optional<TraceRecord>{};
  if (bytes != sizeof(header) || header.magic != kRecordMagic || header.version != 1u ||
      header.reserved != 0u || !valid_kind(header.kind) ||
      header.message_bytes < sizeof(MessageEnvelope) ||
      header.message_bytes > sizeof(ConditionMessage))
    return std::unexpected("Relay trace record header is incompatible or corrupt");
  TraceRecord record{.kind = header.kind, .sequence = header.sequence};
  try {
    record.message.resize(header.message_bytes);
  } catch (...) {
    return std::unexpected("Out of memory reading Relay trace record");
  }
  if (!read_all(file_.get(), record.message.data(), record.message.size()) ||
      checksum(record.message) != header.checksum)
    return std::unexpected("Relay trace record payload is truncated or corrupt");
  MessageEnvelope envelope{};
  std::memcpy(&envelope, record.message.data(), sizeof(envelope));
  if (envelope.magic != kMessageMagic || envelope.version != kMessageVersion ||
      envelope.kind != header.kind || envelope.sequence != header.sequence ||
      envelope.struct_size != record.message.size())
    return std::unexpected("Relay trace record envelope does not match its index");
  return std::optional<TraceRecord>{std::move(record)};
}

} // namespace fe::relay
