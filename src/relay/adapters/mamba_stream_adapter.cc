#include "relay/adapters/mamba_stream_adapter.h"

#include "relay/jobs/state_codec.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <ranges>
#include <utility>

namespace fe::relay {
namespace {

constexpr std::uint32_t kStateVersion = 1u;
constexpr std::size_t kStateHeaderBytes = 24uz;

[[nodiscard]] bool compatible_model(const fe_model_metadata& metadata) noexcept
{
  return (metadata.architecture == FE_ARCH_MAMBA || metadata.architecture == FE_ARCH_MAMBA_FLOW) &&
         metadata.d_model != 0u && metadata.decode_snapshot_bytes != 0u;
}

[[nodiscard]] bool valid_capacity(const fe_model_metadata& metadata,
                                  std::size_t max_tokens) noexcept
{
  if (!compatible_model(metadata) || max_tokens == 0uz ||
      max_tokens > (kMaxJobPayloadBytes / sizeof(std::int32_t)) ||
      !std::in_range<std::size_t>(metadata.d_model) ||
      !std::in_range<std::size_t>(metadata.decode_snapshot_bytes))
    return false;
  if (metadata.d_model > std::numeric_limits<std::size_t>::max() / sizeof(float))
    return false;
  const std::size_t hidden_bytes = static_cast<std::size_t>(metadata.d_model) * sizeof(float);
  const std::size_t snapshot_bytes = static_cast<std::size_t>(metadata.decode_snapshot_bytes);
  if (hidden_bytes > kMaxJobPayloadBytes ||
      snapshot_bytes > std::numeric_limits<std::size_t>::max() - kStateHeaderBytes)
    return false;
  const std::size_t fixed_bytes = kStateHeaderBytes + snapshot_bytes + hidden_bytes;
  return max_tokens <=
         (std::numeric_limits<std::size_t>::max() - fixed_bytes) / sizeof(std::int32_t);
}

} // namespace

std::expected<std::size_t, std::string_view> encode_mamba_stream_request(
    std::span<const std::int32_t> tokens, std::span<std::byte> destination) noexcept
{
  const std::size_t bytes = mamba_stream_request_bytes(tokens.size());
  if (tokens.empty())
    return std::unexpected("Mamba stream request requires at least one token");
  if (bytes == 0uz)
    return std::unexpected("Mamba stream request exceeds the Relay payload limit");
  if (destination.size() < bytes)
    return std::unexpected("Mamba stream request destination is too small");
  StateWriter writer{destination.first(bytes)};
  for (const std::int32_t token : tokens)
    if (!writer.write(token))
      return std::unexpected("Mamba stream request encoding failed");
  return bytes;
}

MambaStreamAdapter::MambaStreamAdapter(Engine engine, fe_model_metadata metadata,
                                       std::size_t max_tokens)
    : engine_{std::move(engine)}, metadata_{metadata}, tokens_(max_tokens),
      hidden_(static_cast<std::size_t>(metadata.d_model)),
      snapshot_bytes_{static_cast<std::size_t>(metadata.decode_snapshot_bytes)}
{
}

std::expected<MambaStreamAdapter, std::string> MambaStreamAdapter::finish_open(
    fe_engine* engine, std::size_t max_tokens) noexcept
{
  Engine owner{engine};
  if (engine == nullptr)
    return std::unexpected(fe_engine_last_error());
  fe_model_metadata metadata{};
  if (fe_engine_model_metadata(engine, &metadata) != 0)
    return std::unexpected(fe_engine_last_error());
  if (!valid_capacity(metadata, max_tokens))
    return std::unexpected("Mamba stream adapter requires compatible state and payload limits");
  try {
    return MambaStreamAdapter{std::move(owner), metadata, max_tokens};
  } catch (const std::exception& error) {
    return std::unexpected("Failed to allocate Mamba stream adapter storage: " +
                           std::string{error.what()});
  } catch (...) {
    return std::unexpected("Failed to allocate Mamba stream adapter storage");
  }
}

std::expected<MambaStreamAdapter, std::string> MambaStreamAdapter::open(
    std::string_view model_path, std::size_t max_tokens, std::optional<unsigned> threads) noexcept
{
  try {
    const std::string path{model_path};
    fe_engine* const engine = threads ? fe_engine_load_with_threads(path.c_str(), *threads)
                                      : fe_engine_load(path.c_str());
    return finish_open(engine, max_tokens);
  } catch (...) {
    return std::unexpected("Failed to open Mamba stream adapter");
  }
}

std::expected<MambaStreamAdapter, std::string> MambaStreamAdapter::open(
    const fe_weights* weights, std::size_t max_tokens, std::optional<unsigned> threads) noexcept
{
  fe_engine* const engine = threads ? fe_engine_create_from_weights(weights, *threads)
                                    : fe_engine_create_from_weights_auto(weights);
  return finish_open(engine, max_tokens);
}

JobRoute MambaStreamAdapter::route() const noexcept
{
  JobRoute result{.kind = JobKind::kStreaming, .state_schema = kMambaStreamStateSchema};
  std::ranges::copy(metadata_.model_digest.bytes, result.model_digest.begin());
  return result;
}

JobDescriptor MambaStreamAdapter::make_descriptor(std::uint64_t session_id,
                                                  std::uint64_t generation, std::size_t token_count,
                                                  std::uint64_t deadline_ns) const noexcept
{
  const JobRoute adapter_route = route();
  return JobDescriptor{.kind = adapter_route.kind,
                       .model_digest = adapter_route.model_digest,
                       .state_schema = adapter_route.state_schema,
                       .session_id = session_id,
                       .generation = generation,
                       .deadline_ns = deadline_ns,
                       .total_work_units = token_count};
}

JobAdapterRegistration MambaStreamAdapter::registration() noexcept
{
  return make_routed_adapter(*this, route(), max_request_bytes(), max_result_bytes());
}

bool MambaStreamAdapter::prepare(std::span<const std::byte> request) noexcept
{
  if (request.empty() || request.size() % sizeof(std::int32_t) != 0uz ||
      request.size() > max_request_bytes())
    return false;
  const std::size_t count = request.size() / sizeof(std::int32_t);
  StateReader reader{request};
  for (std::size_t index{}; index < count; ++index) {
    const auto token = reader.read<std::int32_t>();
    if (!token)
      return false;
    tokens_[index] = *token;
  }
  token_count_ = count;
  completed_ = 0uz;
  prepared_ = true;
  cancelled_ = false;
  return reader.remaining() == 0uz;
}

bool MambaStreamAdapter::begin() noexcept
{
  if (!engine_ || !prepared_ || token_count_ == 0uz)
    return false;
  fe_engine_reset(engine_.get());
  std::ranges::fill(hidden_, 0.0F);
  completed_ = 0uz;
  cancelled_ = false;
  return true;
}

BackendAdvance MambaStreamAdapter::advance(std::size_t budget) noexcept
{
  if (!engine_ || !prepared_ || cancelled_ || budget == 0uz || completed_ >= token_count_)
    return {};
  const std::size_t count = std::min(budget, token_count_ - completed_);
  for (std::size_t index{}; index < count; ++index) {
    if (fe_engine_step(engine_.get(), tokens_[completed_], hidden_.data()) != 0)
      return {};
    ++completed_;
  }
  return {.step = completed_ == token_count_ ? BackendStep::kComplete : BackendStep::kInProgress,
          .completed_work_units = count};
}

std::size_t MambaStreamAdapter::state_bytes() const noexcept
{
  if (!prepared_)
    return 0uz;
  return kStateHeaderBytes + (token_count_ * sizeof(std::int32_t)) +
         (hidden_.size() * sizeof(float)) + snapshot_bytes_;
}

bool MambaStreamAdapter::save_state(std::span<std::byte> destination) const noexcept
{
  if (!engine_ || !prepared_ || destination.size() != state_bytes())
    return false;
  StateWriter writer{destination};
  if (!writer.write(kStateVersion) || !writer.write(std::uint32_t{}) ||
      !writer.write(static_cast<std::uint64_t>(token_count_)) ||
      !writer.write(static_cast<std::uint64_t>(completed_)))
    return false;
  for (std::size_t index{}; index < token_count_; ++index)
    if (!writer.write(tokens_[index]))
      return false;
  for (const float value : hidden_)
    if (!writer.write_float(value))
      return false;
  if (writer.remaining() != snapshot_bytes_)
    return false;
  return fe_engine_export_decode_state(engine_.get(), destination.data() + writer.position(),
                                       snapshot_bytes_) == 0;
}

bool MambaStreamAdapter::load_state(std::span<const std::byte> source) noexcept
{
  StateReader header{source};
  const auto version = header.read<std::uint32_t>();
  const auto reserved = header.read<std::uint32_t>();
  const auto token_count = header.read<std::uint64_t>();
  const auto completed = header.read<std::uint64_t>();
  if (!version || !reserved || !token_count || !completed || *version != kStateVersion ||
      *reserved != 0u || *token_count == 0u || *token_count > tokens_.size() ||
      *completed > *token_count || !std::in_range<std::size_t>(*token_count) ||
      !std::in_range<std::size_t>(*completed))
    return false;
  const std::size_t next_count = static_cast<std::size_t>(*token_count);
  const std::size_t expected = kStateHeaderBytes + (next_count * sizeof(std::int32_t)) +
                               (hidden_.size() * sizeof(float)) + snapshot_bytes_;
  if (source.size() != expected)
    return false;
  const std::size_t snapshot_offset = expected - snapshot_bytes_;
  if (fe_engine_import_decode_state(engine_.get(), source.data() + snapshot_offset,
                                    snapshot_bytes_) != 0)
    return false;

  StateReader reader{source.subspan(kStateHeaderBytes, snapshot_offset - kStateHeaderBytes)};
  for (std::size_t index{}; index < next_count; ++index) {
    const auto token = reader.read<std::int32_t>();
    if (!token)
      return false;
    tokens_[index] = *token;
  }
  for (float& value : hidden_) {
    const auto restored = reader.read_float();
    if (!restored)
      return false;
    value = *restored;
  }
  if (reader.remaining() != 0uz)
    return false;
  token_count_ = next_count;
  completed_ = static_cast<std::size_t>(*completed);
  prepared_ = true;
  cancelled_ = false;
  return true;
}

std::span<const std::byte> MambaStreamAdapter::result() const noexcept
{
  return completed_ == 0uz ? std::span<const std::byte>{} : std::as_bytes(std::span{hidden_});
}

} // namespace fe::relay
