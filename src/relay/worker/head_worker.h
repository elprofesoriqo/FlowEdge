#pragma once

#include "api/engine.h"
#include "relay/protocol/messages.h"

#include <expected>
#include <optional>
#include <string>
#include <string_view>

namespace fe::relay {

enum class WorkerStep : std::uint8_t
{
  kInProgress,
  kComplete,
  kCancelled,
  kError,
};

class HeadWorker
{
public:
  [[nodiscard]] static std::expected<HeadWorker, std::string> open(
      std::string_view model_path, std::optional<unsigned> threads = std::nullopt) noexcept;

  ~HeadWorker();
  HeadWorker(const HeadWorker&) = delete;
  HeadWorker& operator=(const HeadWorker&) = delete;
  HeadWorker(HeadWorker&& other) noexcept;
  HeadWorker& operator=(HeadWorker&& other) noexcept;

  [[nodiscard]] const fe_model_metadata& model_metadata() const noexcept { return model_metadata_; }
  [[nodiscard]] bool busy() const noexcept { return busy_; }
  [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }
  [[nodiscard]] std::uint64_t remaining_nfe() const noexcept { return remaining_nfe_; }
  [[nodiscard]] const std::string& last_error() const noexcept { return last_error_; }

  [[nodiscard]] bool begin(const ConditionMessage& request) noexcept;
  [[nodiscard]] WorkerStep advance(ActionMessage& result) noexcept;
  void cancel_before(std::uint64_t generation) noexcept;

private:
  explicit HeadWorker(fe_engine* engine, fe_model_metadata metadata) noexcept
      : engine_{engine}, model_metadata_{metadata}
  {
  }

  fe_engine* engine_{};
  fe_model_metadata model_metadata_{};
  MessageEnvelope request_envelope_{};
  std::uint64_t generation_{};
  std::uint64_t remaining_nfe_{};
  bool busy_{false};
  std::string last_error_{};
};

} // namespace fe::relay
