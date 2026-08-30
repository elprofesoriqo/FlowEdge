#include "relay/client/relay_client.h"
#include "relay/worker/head_worker.h"

#include <charconv>
#include <chrono>
#include <cstdint>
#include <expected>
#include <iomanip>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using fe::relay::ActionMessage;
using fe::relay::ClientResult;
using fe::relay::HeadWorker;
using fe::relay::RelayClient;
using fe::relay::RelayRequest;

enum class Command : std::uint8_t
{
  kNone,
  kRequest,
  kShutdown,
};

struct Options
{
  Command command{Command::kNone};
  std::string model{};
  std::string condition_shm{"flowedge-relay-conditions"};
  std::string action_shm{"flowedge-relay-actions"};
  std::uint64_t sequence{1u};
  std::uint64_t session_id{1u};
  std::uint64_t generation{1u};
  std::uint64_t deadline_ms{1'000u};
  std::uint64_t timeout_ms{5'000u};
  std::size_t steps{6uz};
  std::uint32_t solver{FE_SOLVER_HEUN};
  float condition_value{0.125f};
  float noise_value{-0.25f};
  std::optional<unsigned> threads{0u};
  bool help{false};
};

[[nodiscard]] std::uint64_t monotonic_ns() noexcept
{
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now().time_since_epoch())
                                        .count());
}

template<typename Number>
[[nodiscard]] bool parse_number(std::string_view text, Number& value) noexcept
{
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc{} && end == text.data() + text.size();
}

void usage(std::ostream& output)
{
  output << "Usage:\n"
            "  flowedge-relayctl request --model FILE [options]\n"
            "  flowedge-relayctl shutdown [options]\n"
            "Options:\n"
            "  --condition-shm NAME  input shared-memory ring name\n"
            "  --action-shm NAME     output shared-memory ring name\n"
            "  --sequence N          request sequence (default 1)\n"
            "  --session N           session identifier (default 1)\n"
            "  --generation N        freshness generation (default 1)\n"
            "  --steps N             solver steps (default 6)\n"
            "  --solver NAME         euler, heun, or rk4 (default heun)\n"
            "  --deadline-ms N       relative request deadline; 0 disables\n"
            "  --timeout-ms N        connect/response timeout (default 5000)\n"
            "  --condition-value F   fill value for the smoke request\n"
            "  --noise-value F       noise fill value for the smoke request\n"
            "  --threads N           metadata loader threads, 0..8\n";
}

[[nodiscard]] std::expected<Options, std::string> parse_options(int argc, char** argv)
{
  Options options{};
  if (argc < 2)
    return std::unexpected("A request or shutdown command is required");
  const std::string_view command{argv[1]};
  if (command == "--help" || command == "-h") {
    options.help = true;
    return options;
  }
  if (command == "request")
    options.command = Command::kRequest;
  else if (command == "shutdown")
    options.command = Command::kShutdown;
  else
    return std::unexpected("Unknown command: " + std::string{command});

  for (int i{2}; i < argc; ++i) {
    const std::string_view argument{argv[i]};
    if (argument == "--help" || argument == "-h") {
      options.help = true;
      return options;
    }
    auto next = [&]() -> std::expected<std::string_view, std::string> {
      if (++i >= argc)
        return std::unexpected("Missing value after " + std::string{argument});
      return std::string_view{argv[i]};
    };
    const auto value = next();
    if (!value)
      return std::unexpected(value.error());
    if (argument == "--model")
      options.model = *value;
    else if (argument == "--condition-shm")
      options.condition_shm = *value;
    else if (argument == "--action-shm")
      options.action_shm = *value;
    else if (argument == "--sequence") {
      if (!parse_number(*value, options.sequence))
        return std::unexpected("Invalid --sequence value");
    } else if (argument == "--session") {
      if (!parse_number(*value, options.session_id))
        return std::unexpected("Invalid --session value");
    } else if (argument == "--generation") {
      if (!parse_number(*value, options.generation))
        return std::unexpected("Invalid --generation value");
    } else if (argument == "--steps") {
      if (!parse_number(*value, options.steps) || options.steps == 0uz)
        return std::unexpected("Invalid --steps value");
    } else if (argument == "--deadline-ms") {
      if (!parse_number(*value, options.deadline_ms))
        return std::unexpected("Invalid --deadline-ms value");
    } else if (argument == "--timeout-ms") {
      if (!parse_number(*value, options.timeout_ms) || options.timeout_ms == 0u)
        return std::unexpected("Invalid --timeout-ms value");
    } else if (argument == "--condition-value") {
      if (!parse_number(*value, options.condition_value))
        return std::unexpected("Invalid --condition-value");
    } else if (argument == "--noise-value") {
      if (!parse_number(*value, options.noise_value))
        return std::unexpected("Invalid --noise-value");
    } else if (argument == "--threads") {
      unsigned threads{};
      if (!parse_number(*value, threads) || threads > 8u)
        return std::unexpected("Invalid --threads value; expected 0..8");
      options.threads = threads;
    } else if (argument == "--solver") {
      if (*value == "euler")
        options.solver = FE_SOLVER_EULER;
      else if (*value == "heun")
        options.solver = FE_SOLVER_HEUN;
      else if (*value == "rk4")
        options.solver = FE_SOLVER_RK4;
      else
        return std::unexpected("Invalid --solver value; expected euler, heun, or rk4");
    } else {
      return std::unexpected("Unknown option: " + std::string{argument});
    }
  }
  if (options.condition_shm.empty() || options.action_shm.empty())
    return std::unexpected("Shared-memory ring names cannot be empty");
  if (options.command == Command::kRequest && options.model.empty())
    return std::unexpected("request requires --model");
  return options;
}

[[nodiscard]] std::expected<fe_model_metadata, std::string> load_metadata(
    const Options& options) noexcept
{
  auto opened = HeadWorker::open(options.model, options.threads);
  if (!opened)
    return std::unexpected(opened.error());
  return opened->model_metadata();
}

[[nodiscard]] std::expected<RelayClient, std::string> connect_until(
    const Options& options, const fe_model_metadata& metadata)
{
  const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds{options.timeout_ms};
  std::string error{};
  do {
    auto connected = RelayClient::connect(options.condition_shm, options.action_shm, metadata);
    if (connected)
      return connected;
    error = connected.error();
    std::this_thread::sleep_for(std::chrono::milliseconds{2});
  } while (std::chrono::steady_clock::now() < end);
  return std::unexpected("Timed out connecting to Relay: " + error);
}

[[nodiscard]] int send_request(const Options& options)
{
  const auto metadata_result = load_metadata(options);
  if (!metadata_result) {
    std::cerr << "flowedge-relayctl: " << metadata_result.error() << '\n';
    return 1;
  }
  const fe_model_metadata metadata = *metadata_result;
  auto connected = connect_until(options, metadata);
  if (!connected) {
    std::cerr << "flowedge-relayctl: " << connected.error() << '\n';
    return 1;
  }
  RelayClient client = std::move(*connected);
  const std::vector<float> condition(metadata.condition_dim, options.condition_value);
  const std::vector<float> noise(metadata.action_dim, options.noise_value);
  const std::uint64_t submitted_at = monotonic_ns();
  const std::uint64_t deadline =
      options.deadline_ms == 0u ? 0u : submitted_at + (options.deadline_ms * 1'000'000u);
  const RelayRequest request{.sequence = options.sequence,
                             .session_id = options.session_id,
                             .timestamp_ns = submitted_at,
                             .deadline_ns = deadline,
                             .generation = options.generation,
                             .solver_steps = options.steps,
                             .solver = options.solver,
                             .condition = condition,
                             .noise = noise};
  const auto timeout =
      std::chrono::steady_clock::now() + std::chrono::milliseconds{options.timeout_ms};
  ClientResult submitted{};
  do {
    submitted = client.try_submit(request);
    if (submitted == ClientResult::kSuccess)
      break;
    if (submitted != ClientResult::kFull) {
      std::cerr << "flowedge-relayctl: submit failed: " << fe::relay::to_string(submitted) << '\n';
      return 1;
    }
    std::this_thread::yield();
  } while (std::chrono::steady_clock::now() < timeout);
  if (submitted != ClientResult::kSuccess) {
    std::cerr << "flowedge-relayctl: timed out submitting request\n";
    return 1;
  }

  ActionMessage action{};
  ClientResult received{};
  do {
    received = client.try_receive(action);
    if (received == ClientResult::kSuccess)
      break;
    if (received != ClientResult::kEmpty) {
      std::cerr << "flowedge-relayctl: receive failed: " << fe::relay::to_string(received) << '\n';
      return 1;
    }
    std::this_thread::yield();
  } while (std::chrono::steady_clock::now() < timeout);
  if (received != ClientResult::kSuccess) {
    std::cerr << "flowedge-relayctl: timed out waiting for action\n";
    return 1;
  }
  if (action.envelope.sequence != options.sequence ||
      action.envelope.session_id != options.session_id) {
    std::cerr << "flowedge-relayctl: received a response for another request\n";
    return 1;
  }

  std::cout << "sequence=" << action.envelope.sequence << " session=" << action.envelope.session_id
            << " generation=" << action.metadata.generation << " status=" << action.metadata.status
            << " action=";
  std::cout << std::setprecision(9);
  for (std::size_t i{0uz}; i < metadata.action_dim; ++i)
    std::cout << (i == 0uz ? "" : ",") << action.action[i];
  std::cout << '\n';
  return action.metadata.status == FE_ACTION_COMPLETE ? 0 : 1;
}

[[nodiscard]] int send_shutdown(const Options& options)
{
  const auto timeout =
      std::chrono::steady_clock::now() + std::chrono::milliseconds{options.timeout_ms};
  std::string error{};
  do {
    const auto sent =
        RelayClient::send_shutdown(options.condition_shm, options.sequence, options.session_id);
    if (sent && *sent == ClientResult::kSuccess)
      return 0;
    if (sent && *sent != ClientResult::kFull) {
      std::cerr << "flowedge-relayctl: shutdown failed: " << fe::relay::to_string(*sent) << '\n';
      return 1;
    }
    if (!sent)
      error = sent.error();
    std::this_thread::sleep_for(std::chrono::milliseconds{2});
  } while (std::chrono::steady_clock::now() < timeout);
  std::cerr << "flowedge-relayctl: timed out sending shutdown";
  if (!error.empty())
    std::cerr << ": " << error;
  std::cerr << '\n';
  return 1;
}

} // namespace

int main(int argc, char** argv)
try {
  const auto parsed = parse_options(argc, argv);
  if (!parsed) {
    std::cerr << "flowedge-relayctl: " << parsed.error() << '\n';
    usage(std::cerr);
    return 2;
  }
  const Options& options = *parsed;
  if (options.help) {
    usage(std::cout);
    return 0;
  }
  return options.command == Command::kRequest ? send_request(options) : send_shutdown(options);
} catch (const std::exception& error) {
  std::cerr << "flowedge-relayctl: " << error.what() << '\n';
  return 1;
}
