#pragma once

#include "api/engine.h"
#include "relay/adapters/routed_adapters.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fe::relay {

inline constexpr std::uint64_t kMambaStreamStateSchema = 0x6d616d62612d7631u; // "mamba-v1"

[[nodiscard]] constexpr std::size_t mamba_stream_request_bytes(std::size_t tokens) noexcept
{
  return tokens <= (kMaxJobPayloadBytes / sizeof(std::int32_t)) ? tokens * sizeof(std::int32_t)
                                                                : 0uz;
}

[[nodiscard]] std::expected<std::size_t, std::string_view> encode_mamba_stream_request(
    std::span<const std::int32_t> tokens, std::span<std::byte> destination) noexcept;

// One instance owns one mutable Core engine and belongs to one Relay execution
// lane. Request, result, and migration storage is fixed during construction.
class MambaStreamAdapter
{
public:
  [[nodiscard]] static std::expected<MambaStreamAdapter, std::string> open(
      std::string_view model_path, std::size_t max_tokens = 512uz,
      std::optional<unsigned> threads = std::nullopt) noexcept;
  [[nodiscard]] static std::expected<MambaStreamAdapter, std::string> open(
      const fe_weights* weights, std::size_t max_tokens = 512uz,
      std::optional<unsigned> threads = std::nullopt) noexcept;

  MambaStreamAdapter(const MambaStreamAdapter&) = delete;
  MambaStreamAdapter& operator=(const MambaStreamAdapter&) = delete;
  MambaStreamAdapter(MambaStreamAdapter&&) noexcept = default;
  MambaStreamAdapter& operator=(MambaStreamAdapter&&) noexcept = default;
  ~MambaStreamAdapter() = default;

  [[nodiscard]] const fe_model_metadata& model_metadata() const noexcept { return metadata_; }
  [[nodiscard]] std::size_t max_tokens() const noexcept { return tokens_.size(); }
  [[nodiscard]] std::size_t max_request_bytes() const noexcept
  {
    return mamba_stream_request_bytes(tokens_.size());
  }
  [[nodiscard]] std::size_t max_result_bytes() const noexcept
  {
    return hidden_.size() * sizeof(float);
  }
  [[nodiscard]] std::size_t max_state_bytes() const noexcept;
  [[nodiscard]] JobRoute route() const noexcept;
  [[nodiscard]] JobDescriptor make_descriptor(std::uint64_t session_id, std::uint64_t generation,
                                              std::size_t token_count,
                                              std::uint64_t deadline_ns = 0u) const noexcept;
  [[nodiscard]] JobAdapterRegistration registration() noexcept;

  [[nodiscard]] bool prepare(std::span<const std::byte> request) noexcept;
  [[nodiscard]] bool begin() noexcept;
  [[nodiscard]] BackendAdvance advance(std::size_t budget) noexcept;
  void cancel() noexcept { cancelled_ = true; }
  [[nodiscard]] std::size_t state_bytes() const noexcept;
  [[nodiscard]] bool save_state(std::span<std::byte> destination) const noexcept;
  [[nodiscard]] bool load_state(std::span<const std::byte> source) noexcept;
  [[nodiscard]] std::span<const std::byte> result() const noexcept;

private:
  struct EngineDeleter
  {
    void operator()(fe_engine* engine) const noexcept { fe_engine_free(engine); }
  };
  using Engine = std::unique_ptr<fe_engine, EngineDeleter>;

  MambaStreamAdapter(Engine engine, fe_model_metadata metadata, std::size_t max_tokens);
  [[nodiscard]] static std::expected<MambaStreamAdapter, std::string> finish_open(
      fe_engine* engine, std::size_t max_tokens) noexcept;

  Engine engine_{};
  fe_model_metadata metadata_{};
  std::vector<std::int32_t> tokens_{};
  std::vector<float> hidden_{};
  std::size_t snapshot_bytes_{};
  std::size_t token_count_{};
  std::size_t completed_{};
  bool prepared_{};
  bool cancelled_{};
};

static_assert(RoutedBackend<MambaStreamAdapter>);

} // namespace fe::relay
