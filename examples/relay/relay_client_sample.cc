#include "api/engine.h"
#include "relay/client/relay_client.h"
#include "relay/protocol/messages.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

struct EngineDeleter
{
  void operator()(fe_engine* engine) const noexcept { fe_engine_free(engine); }
};

[[nodiscard]] std::uint64_t monotonic_ns() noexcept
{
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now().time_since_epoch())
                                        .count());
}

} // namespace

int main(int argc, char** argv)
{
  if (argc < 2 || argc > 4) {
    std::cerr << "usage: relay_client_sample <model.safetensors> [condition-shm] [action-shm]\n";
    return 2;
  }
  const std::string condition_name = argc > 2 ? argv[2] : "flowedge-relay-conditions";
  const std::string action_name = argc > 3 ? argv[3] : "flowedge-relay-actions";

  std::unique_ptr<fe_engine, EngineDeleter> metadata_engine{
      fe_engine_load_with_threads(argv[1], 0u)};
  if (!metadata_engine) {
    std::cerr << "load failed: " << fe_engine_last_error() << '\n';
    return 1;
  }
  fe_model_metadata metadata{};
  if (fe_engine_model_metadata(metadata_engine.get(), &metadata) != 0) {
    std::cerr << "metadata failed: " << fe_engine_last_error() << '\n';
    return 1;
  }

  auto connected = fe::relay::RelayClient::connect(condition_name, action_name, metadata);
  if (!connected) {
    std::cerr << "connect failed: " << connected.error() << '\n';
    return 1;
  }
  fe::relay::RelayClient client = std::move(*connected);
  std::vector<float> condition(metadata.condition_dim, 0.125F);
  std::vector<float> noise(metadata.action_dim, -0.25F);
  const std::uint64_t submitted_at = monotonic_ns();
  const fe::relay::RelayRequest request{
      .sequence = 1u,
      .session_id = 42u,
      .timestamp_ns = submitted_at,
      .deadline_ns = submitted_at + 2'000'000'000u,
      .generation = 1u,
      .solver_steps = 6uz,
      .solver = FE_SOLVER_HEUN,
      .condition = condition,
      .noise = noise,
  };
  const fe::relay::ClientResult submitted = client.try_submit(request);
  if (submitted != fe::relay::ClientResult::kSuccess) {
    std::cerr << "submit failed: " << fe::relay::to_string(submitted) << '\n';
    return 1;
  }

  fe::relay::ActionMessage action{};
  const auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds{5};
  fe::relay::ClientResult received{fe::relay::ClientResult::kEmpty};
  while (received == fe::relay::ClientResult::kEmpty &&
         std::chrono::steady_clock::now() < timeout) {
    received = client.try_receive(action);
    if (received == fe::relay::ClientResult::kEmpty)
      std::this_thread::yield();
  }
  if (received != fe::relay::ClientResult::kSuccess) {
    std::cerr << "receive failed: " << fe::relay::to_string(received) << '\n';
    return 1;
  }

  std::cout << "sequence=" << action.envelope.sequence
            << " generation=" << action.metadata.generation << " status=" << action.metadata.status
            << " outcome=" << fe::relay::to_string(fe::relay::action_code(action));
  if (action.metadata.status == FE_ACTION_COMPLETE && action.metadata.action_dim > 0u)
    std::cout << " action0=" << action.action.front();
  std::cout << '\n';
  return 0;
}
