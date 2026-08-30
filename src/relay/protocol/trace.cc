#include "relay/protocol/trace.h"

#include "protocol/model_identity.h"

#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

namespace fe::relay {
namespace {

constexpr std::array<char, 8> kTraceMagicV1{'F', 'E', 'T', 'R', 'A', 'C', 'E', '1'};
constexpr std::array<char, 8> kTraceMagicV2{'F', 'E', 'T', 'R', 'A', 'C', 'E', '2'};
constexpr std::uint32_t kTraceVersionV1 = 1u;
constexpr std::uint32_t kTraceVersionV2 = 2u;
constexpr std::uint32_t kRecordMagicV1 = 0x31434552u; // "REC1" little-endian
constexpr std::uint32_t kRecordMagicV2 = 0x32434552u; // "REC2" little-endian
constexpr std::size_t kFileHeaderBytes = 16uz;
constexpr std::size_t kRecordHeaderBytes = 32uz;
constexpr std::size_t kEnvelopeBytes = 32uz;
constexpr std::size_t kConditionMetadataBytes = 88uz;
constexpr std::size_t kActionMetadataBytes = 80uz;
constexpr std::size_t kConditionPayloadOffset = kEnvelopeBytes + kConditionMetadataBytes;
constexpr std::size_t kActionPayloadOffset = kEnvelopeBytes + kActionMetadataBytes;

template<std::unsigned_integral Integer>
void write_le(std::span<std::byte> destination, std::size_t offset, Integer value) noexcept
{
  for (std::size_t i{0uz}; i < sizeof(Integer); ++i) {
    destination[offset + i] = static_cast<std::byte>(value & static_cast<Integer>(0xffu));
    value >>= 8u;
  }
}

template<std::unsigned_integral Integer>
[[nodiscard]] Integer read_le(std::span<const std::byte> source, std::size_t offset) noexcept
{
  Integer value{};
  for (std::size_t i{0uz}; i < sizeof(Integer); ++i)
    value |= static_cast<Integer>(std::to_integer<unsigned char>(source[offset + i])) << (8uz * i);
  return value;
}

void write_float(std::span<std::byte> destination, std::size_t offset, float value) noexcept
{
  write_le(destination, offset, std::bit_cast<std::uint32_t>(value));
}

[[nodiscard]] float read_float(std::span<const std::byte> source, std::size_t offset) noexcept
{
  return std::bit_cast<float>(read_le<std::uint32_t>(source, offset));
}

void encode_envelope(std::span<std::byte> destination, const MessageEnvelope& envelope) noexcept
{
  write_le(destination, 0uz, envelope.magic);
  write_le(destination, 4uz, envelope.version);
  write_le(destination, 6uz, static_cast<std::uint16_t>(envelope.kind));
  write_le(destination, 8uz, envelope.struct_size);
  write_le(destination, 12uz, envelope.flags);
  write_le(destination, 16uz, envelope.sequence);
  write_le(destination, 24uz, envelope.session_id);
}

[[nodiscard]] MessageEnvelope decode_envelope(std::span<const std::byte> source) noexcept
{
  MessageEnvelope envelope{};
  envelope.magic = read_le<std::uint32_t>(source, 0uz);
  envelope.version = read_le<std::uint16_t>(source, 4uz);
  envelope.kind = static_cast<MessageKind>(read_le<std::uint16_t>(source, 6uz));
  envelope.struct_size = read_le<std::uint32_t>(source, 8uz);
  envelope.flags = read_le<std::uint32_t>(source, 12uz);
  envelope.sequence = read_le<std::uint64_t>(source, 16uz);
  envelope.session_id = read_le<std::uint64_t>(source, 24uz);
  return envelope;
}

void encode_condition_metadata(std::span<std::byte> destination,
                               const fe_condition_metadata& metadata) noexcept
{
  write_le(destination, 0uz, metadata.struct_size);
  write_le(destination, 4uz, metadata.protocol_version);
  write_le(destination, 8uz, metadata.solver);
  write_le(destination, 12uz, metadata.reserved);
  std::memcpy(destination.data() + 16uz, metadata.model_digest.bytes, FE_MODEL_DIGEST_BYTES);
  write_le(destination, 32uz, metadata.timestamp_ns);
  write_le(destination, 40uz, metadata.deadline_ns);
  write_le(destination, 48uz, metadata.generation);
  write_le(destination, 56uz, metadata.condition_dim);
  write_le(destination, 64uz, metadata.action_dim);
  write_le(destination, 72uz, metadata.solver_steps);
  write_le(destination, 80uz, metadata.remaining_nfe);
}

[[nodiscard]] fe_condition_metadata decode_condition_metadata(
    std::span<const std::byte> source) noexcept
{
  fe_condition_metadata metadata{};
  metadata.struct_size = read_le<std::uint32_t>(source, 0uz);
  metadata.protocol_version = read_le<std::uint32_t>(source, 4uz);
  metadata.solver = read_le<std::uint32_t>(source, 8uz);
  metadata.reserved = read_le<std::uint32_t>(source, 12uz);
  std::memcpy(metadata.model_digest.bytes, source.data() + 16uz, FE_MODEL_DIGEST_BYTES);
  metadata.timestamp_ns = read_le<std::uint64_t>(source, 32uz);
  metadata.deadline_ns = read_le<std::uint64_t>(source, 40uz);
  metadata.generation = read_le<std::uint64_t>(source, 48uz);
  metadata.condition_dim = read_le<std::uint64_t>(source, 56uz);
  metadata.action_dim = read_le<std::uint64_t>(source, 64uz);
  metadata.solver_steps = read_le<std::uint64_t>(source, 72uz);
  metadata.remaining_nfe = read_le<std::uint64_t>(source, 80uz);
  return metadata;
}

void encode_action_metadata(std::span<std::byte> destination,
                            const fe_action_metadata& metadata) noexcept
{
  write_le(destination, 0uz, metadata.struct_size);
  write_le(destination, 4uz, metadata.protocol_version);
  write_le(destination, 8uz, metadata.solver);
  write_le(destination, 12uz, metadata.status);
  std::memcpy(destination.data() + 16uz, metadata.model_digest.bytes, FE_MODEL_DIGEST_BYTES);
  write_le(destination, 32uz, metadata.timestamp_ns);
  write_le(destination, 40uz, metadata.deadline_ns);
  write_le(destination, 48uz, metadata.generation);
  write_le(destination, 56uz, metadata.condition_dim);
  write_le(destination, 64uz, metadata.action_dim);
  write_le(destination, 72uz, metadata.remaining_nfe);
}

[[nodiscard]] fe_action_metadata decode_action_metadata(std::span<const std::byte> source) noexcept
{
  fe_action_metadata metadata{};
  metadata.struct_size = read_le<std::uint32_t>(source, 0uz);
  metadata.protocol_version = read_le<std::uint32_t>(source, 4uz);
  metadata.solver = read_le<std::uint32_t>(source, 8uz);
  metadata.status = read_le<std::uint32_t>(source, 12uz);
  std::memcpy(metadata.model_digest.bytes, source.data() + 16uz, FE_MODEL_DIGEST_BYTES);
  metadata.timestamp_ns = read_le<std::uint64_t>(source, 32uz);
  metadata.deadline_ns = read_le<std::uint64_t>(source, 40uz);
  metadata.generation = read_le<std::uint64_t>(source, 48uz);
  metadata.condition_dim = read_le<std::uint64_t>(source, 56uz);
  metadata.action_dim = read_le<std::uint64_t>(source, 64uz);
  metadata.remaining_nfe = read_le<std::uint64_t>(source, 72uz);
  return metadata;
}

[[nodiscard]] constexpr bool valid_kind(MessageKind kind) noexcept
{
  return kind == MessageKind::kCondition || kind == MessageKind::kAction ||
         kind == MessageKind::kShutdown;
}

[[nodiscard]] std::expected<std::size_t, std::string> encode_message(
    std::span<const std::byte> source, std::span<std::byte> destination) noexcept
{
  if (source.size() < kEnvelopeBytes || source.size() > destination.size())
    return std::unexpected("Relay trace message size is invalid");
  MessageEnvelope envelope{};
  std::memcpy(&envelope, source.data(), sizeof(envelope));
  if (envelope.magic != kMessageMagic || envelope.version != kMessageVersion ||
      !valid_kind(envelope.kind) || envelope.struct_size != source.size())
    return std::unexpected("Relay trace message envelope is invalid");

  encode_envelope(destination, envelope);
  if (envelope.kind == MessageKind::kCondition) {
    if (source.size() > sizeof(ConditionMessage))
      return std::unexpected("Relay trace condition message is too large");
    ConditionMessage message{};
    std::memcpy(&message, source.data(), source.size());
    if (validate(message) != ProtocolResult::kSuccess || wire_size(message) != source.size())
      return std::unexpected("Relay trace condition message is invalid");
    encode_condition_metadata(destination.subspan(kEnvelopeBytes), message.metadata);
    for (std::size_t i{0uz}; i < message.condition_values().size() + message.noise_values().size();
         ++i)
      write_float(destination, kConditionPayloadOffset + (i * sizeof(float)), message.payload[i]);
  } else if (envelope.kind == MessageKind::kAction) {
    if (source.size() > sizeof(ActionMessage))
      return std::unexpected("Relay trace action message is too large");
    ActionMessage message{};
    std::memcpy(&message, source.data(), source.size());
    if (validate(message) != ProtocolResult::kSuccess || wire_size(message) != source.size())
      return std::unexpected("Relay trace action message is invalid");
    encode_action_metadata(destination.subspan(kEnvelopeBytes), message.metadata);
    for (std::size_t i{0uz}; i < message.metadata.action_dim; ++i)
      write_float(destination, kActionPayloadOffset + (i * sizeof(float)), message.action[i]);
  } else {
    if (source.size() != sizeof(ControlMessage))
      return std::unexpected("Relay trace control message size is invalid");
    ControlMessage message{};
    std::memcpy(&message, source.data(), sizeof(message));
    if (!valid_envelope(message.envelope, MessageKind::kShutdown, sizeof(message)))
      return std::unexpected("Relay trace control message is invalid");
    write_le(destination, kEnvelopeBytes, message.reason);
  }
  return source.size();
}

[[nodiscard]] std::expected<std::size_t, std::string> decode_message(
    std::span<const std::byte> source, std::span<std::byte> destination) noexcept
{
  if (source.size() < kEnvelopeBytes || source.size() > destination.size())
    return std::unexpected("Relay trace encoded message size is invalid");
  const MessageEnvelope envelope = decode_envelope(source);
  if (envelope.magic != kMessageMagic || envelope.version != kMessageVersion ||
      !valid_kind(envelope.kind) || envelope.struct_size != source.size())
    return std::unexpected("Relay trace encoded envelope is invalid");

  if (envelope.kind == MessageKind::kCondition) {
    if (source.size() < kConditionPayloadOffset ||
        (source.size() - kConditionPayloadOffset) % sizeof(float) != 0uz)
      return std::unexpected("Relay trace encoded condition size is invalid");
    ConditionMessage message{};
    message.envelope = envelope;
    message.metadata = decode_condition_metadata(source.subspan(kEnvelopeBytes));
    if (wire_size(message) != source.size())
      return std::unexpected("Relay trace encoded condition dimensions are invalid");
    for (std::size_t i{0uz}; i < message.condition_values().size() + message.noise_values().size();
         ++i)
      message.payload[i] = read_float(source, kConditionPayloadOffset + (i * sizeof(float)));
    if (validate(message) != ProtocolResult::kSuccess)
      return std::unexpected("Relay trace decoded condition is invalid");
    std::memcpy(destination.data(), &message, source.size());
  } else if (envelope.kind == MessageKind::kAction) {
    if (source.size() < kActionPayloadOffset ||
        (source.size() - kActionPayloadOffset) % sizeof(float) != 0uz)
      return std::unexpected("Relay trace encoded action size is invalid");
    ActionMessage message{};
    message.envelope = envelope;
    message.metadata = decode_action_metadata(source.subspan(kEnvelopeBytes));
    if (wire_size(message) != source.size())
      return std::unexpected("Relay trace encoded action dimensions are invalid");
    for (std::size_t i{0uz}; i < message.metadata.action_dim; ++i)
      message.action[i] = read_float(source, kActionPayloadOffset + (i * sizeof(float)));
    if (validate(message) != ProtocolResult::kSuccess)
      return std::unexpected("Relay trace decoded action is invalid");
    std::memcpy(destination.data(), &message, source.size());
  } else {
    if (source.size() != sizeof(ControlMessage))
      return std::unexpected("Relay trace encoded control size is invalid");
    ControlMessage message{};
    message.envelope = envelope;
    message.reason = read_le<std::uint64_t>(source, kEnvelopeBytes);
    if (!valid_envelope(message.envelope, MessageKind::kShutdown, sizeof(message)))
      return std::unexpected("Relay trace decoded control message is invalid");
    std::memcpy(destination.data(), &message, sizeof(message));
  }
  return source.size();
}

[[nodiscard]] std::uint64_t checksum(std::span<const std::byte> bytes) noexcept
{
  const ModelDigest digest = fingerprint_bytes(bytes);
  std::uint64_t value{};
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
  std::array<std::byte, kFileHeaderBytes> header{};
  std::ranges::copy(std::as_bytes(std::span{kTraceMagicV2}), header.begin());
  write_le(std::span{header}, 8uz, kTraceVersionV2);
  write_le(std::span{header}, 12uz, static_cast<std::uint32_t>(header.size()));
  if (!write_all(file.get(), header.data(), header.size()))
    return std::unexpected("Failed to write Relay trace header");
  return TraceWriter{file.release()};
}

std::expected<void, std::string> TraceWriter::append(std::span<const std::byte> message) noexcept
{
  if (file_ == nullptr || message.size() > std::numeric_limits<std::uint32_t>::max())
    return std::unexpected("Relay trace message is invalid");
  const auto encoded_size = encode_message(message, encoded_);
  if (!encoded_size)
    return std::unexpected(encoded_size.error());
  const std::span<const std::byte> encoded{encoded_.data(), *encoded_size};
  const MessageEnvelope envelope = decode_envelope(encoded);
  std::array<std::byte, kRecordHeaderBytes> record{};
  write_le(std::span{record}, 0uz, kRecordMagicV2);
  write_le(std::span{record}, 4uz, static_cast<std::uint16_t>(kTraceVersionV2));
  write_le(std::span{record}, 6uz, static_cast<std::uint16_t>(envelope.kind));
  write_le(std::span{record}, 8uz, static_cast<std::uint32_t>(encoded.size()));
  write_le(std::span{record}, 12uz, 0u);
  write_le(std::span{record}, 16uz, envelope.sequence);
  write_le(std::span{record}, 24uz, checksum(encoded));
  if (!write_all(file_.get(), record.data(), record.size()) ||
      !write_all(file_.get(), encoded.data(), encoded.size()))
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
  std::array<std::byte, kFileHeaderBytes> header{};
  if (!read_all(file.get(), header.data(), header.size()))
    return std::unexpected("Relay trace header is truncated");
  const auto magic = std::span{reinterpret_cast<const char*>(header.data()), 8uz};
  const std::uint32_t version = read_le<std::uint32_t>(header, 8uz);
  const std::uint32_t header_bytes = read_le<std::uint32_t>(header, 12uz);
  const bool v2 = std::ranges::equal(magic, kTraceMagicV2) && version == kTraceVersionV2;
  const bool v1 = std::ranges::equal(magic, kTraceMagicV1) && version == kTraceVersionV1;
  if ((!v1 && !v2) || header_bytes != header.size())
    return std::unexpected("Relay trace header is incompatible or corrupt");
  if (v1 && std::endian::native != std::endian::little)
    return std::unexpected("Native-endian Relay trace v1 is unsupported on this host");
  return TraceReader{file.release(), version};
}

std::expected<std::optional<TraceRecord>, std::string> TraceReader::next() noexcept
{
  if (file_ == nullptr)
    return std::unexpected("Relay trace reader is closed");
  std::array<std::byte, kRecordHeaderBytes> header{};
  const std::size_t bytes = std::fread(header.data(), 1uz, header.size(), file_.get());
  if (bytes == 0uz && std::feof(file_.get()) != 0)
    return std::optional<TraceRecord>{};
  const std::uint32_t expected_magic =
      version_ == kTraceVersionV2 ? kRecordMagicV2 : kRecordMagicV1;
  const std::uint32_t record_magic = read_le<std::uint32_t>(header, 0uz);
  const std::uint16_t record_version = read_le<std::uint16_t>(header, 4uz);
  const MessageKind kind = static_cast<MessageKind>(read_le<std::uint16_t>(header, 6uz));
  const std::uint32_t message_bytes = read_le<std::uint32_t>(header, 8uz);
  const std::uint32_t reserved = read_le<std::uint32_t>(header, 12uz);
  const std::uint64_t sequence = read_le<std::uint64_t>(header, 16uz);
  const std::uint64_t expected_checksum = read_le<std::uint64_t>(header, 24uz);
  if (bytes != header.size() || record_magic != expected_magic || record_version != version_ ||
      reserved != 0u || !valid_kind(kind) || message_bytes < kEnvelopeBytes ||
      message_bytes > sizeof(ConditionMessage))
    return std::unexpected("Relay trace record header is incompatible or corrupt");

  std::vector<std::byte> encoded{};
  try {
    encoded.resize(message_bytes);
  } catch (...) {
    return std::unexpected("Out of memory reading Relay trace record");
  }
  if (!read_all(file_.get(), encoded.data(), encoded.size()) ||
      checksum(encoded) != expected_checksum)
    return std::unexpected("Relay trace record payload is truncated or corrupt");

  TraceRecord record{.kind = kind, .sequence = sequence};
  if (version_ == kTraceVersionV1) {
    record.message = std::move(encoded);
  } else {
    try {
      record.message.resize(message_bytes);
    } catch (...) {
      return std::unexpected("Out of memory decoding Relay trace record");
    }
    const auto decoded = decode_message(encoded, record.message);
    if (!decoded)
      return std::unexpected(decoded.error());
  }

  MessageEnvelope envelope{};
  std::memcpy(&envelope, record.message.data(), sizeof(envelope));
  if (envelope.magic != kMessageMagic || envelope.version != kMessageVersion ||
      envelope.kind != kind || envelope.sequence != sequence ||
      envelope.struct_size != record.message.size())
    return std::unexpected("Relay trace record envelope does not match its index");
  return std::optional<TraceRecord>{std::move(record)};
}

} // namespace fe::relay
