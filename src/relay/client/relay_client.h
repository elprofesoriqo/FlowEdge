#pragma once

#include "relay/client/client_result.h"
#include "relay/protocol/messages.h"
#include "relay/shared_memory/shared_memory_ring.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace fe::relay {

struct RelayRequest
{
  std::uint64_t sequence{};
  std::uint64_t session_id{};
  std::uint64_t timestamp_ns{};
  std::uint64_t deadline_ns{};
  std::uint64_t generation{};
  std::size_t solver_steps{6uz};
  std::uint32_t solver{1u};
  std::span<const float> condition{};
  std::span<const float> noise{};
};

// Allocation-free SPSC endpoint for one Relay producer/action consumer. A
// client is intentionally move-only and not thread-safe: each shared-memory
// ring has exactly one producer and one consumer.
class RelayClient
{
public:
  [[nodiscard]] static std::expected<RelayClient, std::string> create(
      std::string condition_name, std::string action_name, const fe_model_metadata& model,
      std::uint32_t capacity = 32u) noexcept;
  [[nodiscard]] static std::expected<RelayClient, std::string> connect(
      std::string condition_name, std::string action_name, const fe_model_metadata& model) noexcept;

  [[nodiscard]] static std::expected<ClientResult, std::string> send_shutdown(
      std::string condition_name, std::uint64_t sequence, std::uint64_t session_id,
      std::uint64_t reason = 0u) noexcept;

  RelayClient(const RelayClient&) = delete;
  RelayClient& operator=(const RelayClient&) = delete;
  RelayClient(RelayClient&&) noexcept = default;
  RelayClient& operator=(RelayClient&&) noexcept = default;

  [[nodiscard]] ClientResult try_submit(const RelayRequest& request) noexcept;
  [[nodiscard]] ClientResult try_receive(ActionMessage& action) noexcept;
  [[nodiscard]] ClientResult try_shutdown(std::uint64_t sequence, std::uint64_t session_id,
                                          std::uint64_t reason = 0u) noexcept;

  [[nodiscard]] const fe_model_metadata& model_metadata() const noexcept { return model_; }
  [[nodiscard]] std::uint64_t pending_conditions() const noexcept { return conditions_.size(); }
  [[nodiscard]] std::uint64_t pending_actions() const noexcept { return actions_.size(); }

private:
  RelayClient(SharedMemoryRing conditions, SharedMemoryRing actions,
              fe_model_metadata model) noexcept
      : conditions_{std::move(conditions)}, actions_{std::move(actions)}, model_{model}
  {
  }

  [[nodiscard]] static std::expected<void, std::string> validate_model(
      const fe_model_metadata& model) noexcept;
  [[nodiscard]] static ClientResult map_ring_result(RingResult result) noexcept;

  SharedMemoryRing conditions_{};
  SharedMemoryRing actions_{};
  fe_model_metadata model_{};
  ConditionMessage outbound_{};
};

} // namespace fe::relay
