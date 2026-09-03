#include "relay/adapters/mamba_stream_adapter.h"
#include "relay/client/job_client.h"
#include "relay/client/job_control_client.h"
#include "relay/jobs/state_codec.h"

#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <expected>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using namespace fe::relay;

enum class Command : std::uint8_t
{
  kNone,
  kMamba,
  kStatus,
  kDrain,
  kResume,
  kShutdown,
};

struct Options
{
  Command command{Command::kNone};
  std::string model{};
  std::string request_shm{"flowedge-job-requests"};
  std::string result_shm{"flowedge-job-results"};
  std::string control_shm{"flowedge-job-control"};
  std::string status_shm{"flowedge-job-status"};
  std::vector<std::int32_t> tokens{1, 2, 3, 4};
  std::uint64_t sequence{1u};
  std::uint64_t session_id{1u};
  std::uint64_t generation{1u};
  std::uint64_t deadline_ms{1'000u};
  std::uint64_t timeout_ms{5'000u};
  std::uint32_t worker_index{kAllJobWorkers};
  bool help{};
};

template<typename Integer>
[[nodiscard]] bool parse_integer(std::string_view text, Integer& value) noexcept
{
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc{} && end == text.data() + text.size();
}

[[nodiscard]] std::expected<std::vector<std::int32_t>, std::string> parse_tokens(
    std::string_view text)
{
  std::vector<std::int32_t> tokens{};
  while (!text.empty()) {
    const std::size_t comma = text.find(',');
    const std::string_view token = text.substr(0uz, comma);
    std::int32_t value{};
    if (token.empty() || !parse_integer(token, value))
      return std::unexpected("Invalid --tokens value");
    tokens.push_back(value);
    if (comma == std::string_view::npos)
      break;
    text.remove_prefix(comma + 1uz);
  }
  if (tokens.empty() || mamba_stream_request_bytes(tokens.size()) == 0uz)
    return std::unexpected("--tokens exceeds the job payload capacity");
  return tokens;
}

void usage(std::ostream& output)
{
  output << "Usage:\n"
            "  flowedge-jobctl mamba --model FILE [options]\n"
            "  flowedge-jobctl status [options]\n"
            "  flowedge-jobctl drain --worker N [options]\n"
            "  flowedge-jobctl resume --worker N [options]\n"
            "  flowedge-jobctl shutdown [options]\n"
            "Options:\n"
            "  --request-shm NAME   job request ring\n"
            "  --result-shm NAME    job result ring\n"
            "  --control-shm NAME   administration request ring\n"
            "  --status-shm NAME    administration response ring\n"
            "  --tokens CSV         streaming token IDs (default 1,2,3,4)\n"
            "  --sequence N         request sequence\n"
            "  --session N          session identifier\n"
            "  --generation N       freshness generation\n"
            "  --deadline-ms N      relative deadline; 0 disables\n"
            "  --worker N           drain/resume lane index\n"
            "  --timeout-ms N       connect/response timeout\n";
}

[[nodiscard]] std::expected<Options, std::string> parse_options(int argc, char** argv)
{
  Options options{};
  if (argc < 2)
    return std::unexpected("A command is required");
  const std::string_view command{argv[1]};
  if (command == "--help" || command == "-h") {
    options.help = true;
    return options;
  }
  if (command == "mamba")
    options.command = Command::kMamba;
  else if (command == "status")
    options.command = Command::kStatus;
  else if (command == "drain")
    options.command = Command::kDrain;
  else if (command == "resume")
    options.command = Command::kResume;
  else if (command == "shutdown")
    options.command = Command::kShutdown;
  else
    return std::unexpected("Unknown command: " + std::string{command});

  for (int index{2}; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "--help" || argument == "-h") {
      options.help = true;
      return options;
    }
    if (++index >= argc)
      return std::unexpected("Missing value after " + std::string{argument});
    const std::string_view value{argv[index]};
    if (argument == "--model")
      options.model = value;
    else if (argument == "--request-shm")
      options.request_shm = value;
    else if (argument == "--result-shm")
      options.result_shm = value;
    else if (argument == "--control-shm")
      options.control_shm = value;
    else if (argument == "--status-shm")
      options.status_shm = value;
    else if (argument == "--tokens") {
      auto tokens = parse_tokens(value);
      if (!tokens)
        return std::unexpected(tokens.error());
      options.tokens = std::move(*tokens);
    } else if (argument == "--sequence") {
      if (!parse_integer(value, options.sequence))
        return std::unexpected("Invalid --sequence value");
    } else if (argument == "--session") {
      if (!parse_integer(value, options.session_id))
        return std::unexpected("Invalid --session value");
    } else if (argument == "--generation") {
      if (!parse_integer(value, options.generation))
        return std::unexpected("Invalid --generation value");
    } else if (argument == "--deadline-ms") {
      if (!parse_integer(value, options.deadline_ms))
        return std::unexpected("Invalid --deadline-ms value");
    } else if (argument == "--timeout-ms") {
      if (!parse_integer(value, options.timeout_ms) || options.timeout_ms == 0u)
        return std::unexpected("Invalid --timeout-ms value");
    } else if (argument == "--worker") {
      if (!parse_integer(value, options.worker_index))
        return std::unexpected("Invalid --worker value");
    } else {
      return std::unexpected("Unknown option: " + std::string{argument});
    }
  }
  if (options.command == Command::kMamba && options.model.empty())
    return std::unexpected("mamba requires --model");
  if ((options.command == Command::kDrain || options.command == Command::kResume) &&
      options.worker_index == kAllJobWorkers)
    return std::unexpected("drain and resume require --worker");
  if (options.request_shm.empty() || options.result_shm.empty() || options.control_shm.empty() ||
      options.status_shm.empty())
    return std::unexpected("Shared-memory ring names cannot be empty");
  return options;
}

[[nodiscard]] std::uint64_t monotonic_ns() noexcept
{
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now().time_since_epoch())
                                        .count());
}

template<typename Client, typename Connect>
[[nodiscard]] std::expected<Client, std::string> connect_until(Connect connect,
                                                               std::uint64_t timeout_ms)
{
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{timeout_ms};
  std::string error{};
  do {
    auto connected = connect();
    if (connected)
      return connected;
    error = connected.error();
    std::this_thread::sleep_for(std::chrono::milliseconds{2});
  } while (std::chrono::steady_clock::now() < deadline);
  return std::unexpected("Timed out connecting: " + error);
}

[[nodiscard]] int send_mamba(const Options& options)
{
  auto adapter = MambaStreamAdapter::open(options.model, options.tokens.size(), 0u);
  if (!adapter) {
    std::cerr << "flowedge-jobctl: " << adapter.error() << '\n';
    return 1;
  }
  auto client = connect_until<JobClient>(
      [&] { return JobClient::connect(options.request_shm, options.result_shm); },
      options.timeout_ms);
  if (!client) {
    std::cerr << "flowedge-jobctl: " << client.error() << '\n';
    return 1;
  }

  std::array<std::byte, kMaxJobPayloadBytes> payload{};
  const auto encoded = encode_mamba_stream_request(options.tokens, payload);
  if (!encoded)
    return 1;
  const std::uint64_t now = monotonic_ns();
  if (options.deadline_ms > (std::numeric_limits<std::uint64_t>::max() - now) / 1'000'000u) {
    std::cerr << "flowedge-jobctl: deadline overflows monotonic time\n";
    return 1;
  }
  const std::uint64_t deadline =
      options.deadline_ms == 0u ? 0u : now + (options.deadline_ms * 1'000'000u);
  const JobDescriptor descriptor = adapter->make_descriptor(options.session_id, options.generation,
                                                            options.tokens.size(), deadline);
  auto request = std::make_unique<JobRequestMessage>();
  if (make_job_request(*request, options.sequence, descriptor, now,
                       std::span{payload}.first(*encoded)) != ProtocolResult::kSuccess)
    return 1;

  const auto timeout =
      std::chrono::steady_clock::now() + std::chrono::milliseconds{options.timeout_ms};
  ClientResult submitted{ClientResult::kFull};
  do {
    submitted = client->try_submit(*request);
    if (submitted != ClientResult::kFull)
      break;
    std::this_thread::yield();
  } while (std::chrono::steady_clock::now() < timeout);
  if (submitted != ClientResult::kSuccess) {
    std::cerr << "flowedge-jobctl: submit failed: " << to_string(submitted) << '\n';
    return 1;
  }

  auto result = std::make_unique<JobResultMessage>();
  ClientResult received{ClientResult::kEmpty};
  do {
    received = client->try_receive(*result);
    if (received != ClientResult::kEmpty)
      break;
    std::this_thread::yield();
  } while (std::chrono::steady_clock::now() < timeout);
  if (received != ClientResult::kSuccess) {
    std::cerr << "flowedge-jobctl: receive failed: " << to_string(received) << '\n';
    return 1;
  }
  float hidden0{};
  if (result->payload_values().size() >= sizeof(hidden0))
    std::memcpy(&hidden0, result->payload_values().data(), sizeof(hidden0));
  std::cout << "sequence=" << result->envelope.sequence
            << " outcome=" << to_string(job_result_code(*result))
            << " completed=" << result->metadata.progress.completed_work_units
            << " result_bytes=" << result->metadata.payload_bytes << " hidden0=" << hidden0 << '\n';
  return job_result_code(*result) == JobResultCode::kComplete ? 0 : 1;
}

[[nodiscard]] JobControlOperation operation(Command command) noexcept
{
  switch (command) {
  case Command::kStatus:
    return JobControlOperation::kStatus;
  case Command::kDrain:
    return JobControlOperation::kDrainWorker;
  case Command::kResume:
    return JobControlOperation::kResumeWorker;
  case Command::kShutdown:
    return JobControlOperation::kShutdown;
  case Command::kNone:
  case Command::kMamba:
    break;
  }
  return JobControlOperation::kStatus;
}

[[nodiscard]] int send_control(const Options& options)
{
  auto client = connect_until<JobControlClient>(
      [&] { return JobControlClient::connect(options.control_shm, options.status_shm); },
      options.timeout_ms);
  if (!client) {
    std::cerr << "flowedge-jobctl: " << client.error() << '\n';
    return 1;
  }
  const JobControlRequest request =
      make_job_control_request(options.sequence, options.session_id, operation(options.command),
                               options.worker_index);
  const auto timeout =
      std::chrono::steady_clock::now() + std::chrono::milliseconds{options.timeout_ms};
  ClientResult submitted{ClientResult::kFull};
  do {
    submitted = client->try_submit(request);
    if (submitted != ClientResult::kFull)
      break;
    std::this_thread::yield();
  } while (std::chrono::steady_clock::now() < timeout);
  if (submitted != ClientResult::kSuccess) {
    std::cerr << "flowedge-jobctl: control submit failed: " << to_string(submitted) << '\n';
    return 1;
  }
  JobControlResponse response{};
  ClientResult received{ClientResult::kEmpty};
  do {
    received = client->try_receive(response);
    if (received != ClientResult::kEmpty)
      break;
    std::this_thread::yield();
  } while (std::chrono::steady_clock::now() < timeout);
  if (received != ClientResult::kSuccess || response.envelope.sequence != options.sequence ||
      response.metadata.operation != request.metadata.operation) {
    std::cerr << "flowedge-jobctl: control response failed\n";
    return 1;
  }
  const JobControlStatus& status = response.metadata.status;
  std::cout << "operation=" << to_string(response.metadata.operation)
            << " code=" << to_string(response.metadata.code) << " workers=" << status.worker_count
            << " accepting=" << status.accepting_workers << " busy=" << status.busy_workers
            << " queued=" << status.queued_jobs << " draining_mask=" << status.draining_mask
            << " drained_mask=" << status.drained_mask << " accepted=" << status.requests_accepted
            << " rejected=" << status.requests_rejected << " published=" << status.results_published
            << " worker_failures=" << status.worker_failures << '\n';
  return response.metadata.code == JobControlCode::kSuccess ? 0 : 1;
}

} // namespace

int main(int argc, char** argv)
try {
  const auto parsed = parse_options(argc, argv);
  if (!parsed) {
    std::cerr << "flowedge-jobctl: " << parsed.error() << '\n';
    usage(std::cerr);
    return 2;
  }
  if (parsed->help) {
    usage(std::cout);
    return 0;
  }
  return parsed->command == Command::kMamba ? send_mamba(*parsed) : send_control(*parsed);
} catch (const std::exception& error) {
  std::cerr << "flowedge-jobctl: " << error.what() << '\n';
  return 1;
} catch (...) {
  std::cerr << "flowedge-jobctl: unknown failure\n";
  return 1;
}
