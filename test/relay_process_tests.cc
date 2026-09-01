#include "relay/client/job_client.h"
#include "relay/client/relay_client.h"
#include "relay/jobs/state_codec.h"
#include "relay/worker/head_worker.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <csignal>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

using namespace fe::relay;

[[nodiscard]] std::uint64_t monotonic_ns() noexcept
{
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now().time_since_epoch())
                                        .count());
}

[[nodiscard]] std::string unique_name(std::string_view prefix)
{
  return std::string{prefix} + '-' + std::to_string(monotonic_ns());
}

class ChildDaemon
{
public:
  [[nodiscard]] static std::optional<ChildDaemon> start(const std::filesystem::path& executable,
                                                        const std::filesystem::path& model,
                                                        const std::string& condition_name,
                                                        const std::string& action_name)
  {
#ifdef _WIN32
    const auto quote = [](std::wstring_view argument) {
      std::wstring result{L"\""};
      std::size_t slashes{};
      for (const wchar_t character : argument) {
        if (character == L'\\') {
          ++slashes;
        } else {
          if (character == L'\"')
            result.append((slashes * 2uz) + 1uz, L'\\');
          else
            result.append(slashes, L'\\');
          slashes = 0uz;
          result.push_back(character);
        }
      }
      result.append(slashes * 2uz, L'\\');
      result.push_back(L'\"');
      return result;
    };
    const auto widen_ascii = [](std::string_view value) {
      return std::wstring{value.begin(), value.end()};
    };
    std::wstring command = quote(executable.wstring()) + L" --model " + quote(model.wstring()) +
                           L" --condition-shm " + quote(widen_ascii(condition_name)) +
                           L" --action-shm " + quote(widen_ascii(action_name)) +
                           L" --capacity 8 --workers 2 --threads 0 --nfe-ns 1000000000 --create";
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE, 0u, nullptr, nullptr,
                       &startup, &process) == FALSE)
      return std::nullopt;
    CloseHandle(process.hThread);
    return ChildDaemon{process.hProcess};
#else
    const pid_t process = fork();
    if (process < 0)
      return std::nullopt;
    if (process == 0) {
      execl(executable.c_str(), executable.c_str(), "--model", model.c_str(), "--condition-shm",
            condition_name.c_str(), "--action-shm", action_name.c_str(), "--capacity", "8",
            "--workers", "2", "--threads", "0", "--nfe-ns", "1000000000", "--create",
            static_cast<char*>(nullptr));
      _exit(127);
    }
    return ChildDaemon{process};
#endif
  }

  [[nodiscard]] static std::optional<ChildDaemon> start_job_service(
      const std::filesystem::path& executable, const std::string& request_name,
      const std::string& result_name)
  {
#ifdef _WIN32
    const auto quote = [](std::wstring_view argument) {
      return L"\"" + std::wstring{argument} + L"\"";
    };
    const auto widen_ascii = [](std::string_view value) {
      return std::wstring{value.begin(), value.end()};
    };
    std::wstring command = quote(executable.wstring()) + L" " + quote(widen_ascii(request_name)) +
                           L" " + quote(widen_ascii(result_name));
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE, 0u, nullptr, nullptr,
                       &startup, &process) == FALSE)
      return std::nullopt;
    CloseHandle(process.hThread);
    return ChildDaemon{process.hProcess};
#else
    const pid_t process = fork();
    if (process < 0)
      return std::nullopt;
    if (process == 0) {
      execl(executable.c_str(), executable.c_str(), request_name.c_str(), result_name.c_str(),
            static_cast<char*>(nullptr));
      _exit(127);
    }
    return ChildDaemon{process};
#endif
  }

  ~ChildDaemon()
  {
    if (!running_)
      return;
#ifdef _WIN32
    TerminateProcess(process_, 1u);
    WaitForSingleObject(process_, 2'000u);
    CloseHandle(process_);
#else
    kill(process_, SIGTERM);
    int status{};
    waitpid(process_, &status, 0);
#endif
  }

  ChildDaemon(const ChildDaemon&) = delete;
  ChildDaemon& operator=(const ChildDaemon&) = delete;

  ChildDaemon(ChildDaemon&& other) noexcept
#ifdef _WIN32
      : process_{std::exchange(other.process_, nullptr)},
        running_{std::exchange(other.running_, false)}
#else
      : process_{std::exchange(other.process_, -1)}, running_{std::exchange(other.running_, false)}
#endif
  {
  }

  [[nodiscard]] std::optional<int> wait_for(std::chrono::milliseconds timeout) noexcept
  {
#ifdef _WIN32
    if (WaitForSingleObject(process_, static_cast<DWORD>(timeout.count())) != WAIT_OBJECT_0)
      return std::nullopt;
    DWORD exit_code{};
    if (GetExitCodeProcess(process_, &exit_code) == FALSE)
      return std::nullopt;
    running_ = false;
    CloseHandle(process_);
    process_ = nullptr;
    return static_cast<int>(exit_code);
#else
    const auto end = std::chrono::steady_clock::now() + timeout;
    do {
      int status{};
      const pid_t waited = waitpid(process_, &status, WNOHANG);
      if (waited == process_) {
        running_ = false;
        return WIFEXITED(status) ? std::optional<int>{WEXITSTATUS(status)}
                                 : std::optional<int>{128};
      }
      if (waited < 0)
        return std::nullopt;
      std::this_thread::sleep_for(std::chrono::milliseconds{2});
    } while (std::chrono::steady_clock::now() < end);
    return std::nullopt;
#endif
  }

private:
#ifdef _WIN32
  explicit ChildDaemon(HANDLE process) noexcept : process_{process} {}
  HANDLE process_{};
#else
  explicit ChildDaemon(pid_t process) noexcept : process_{process} {}
  pid_t process_{-1};
#endif
  bool running_{true};
};

TEST(RelayProcess, ExchangesRequestWithDaemonAcrossProcessBoundary)
{
  const std::filesystem::path model =
      std::filesystem::path{FLOWEDGE_SOURCE_DIR} / "models" / "mamba_flow.safetensors";
  if (!std::filesystem::exists(model))
    GTEST_SKIP() << "models/mamba_flow.safetensors is not available";

  fe_model_metadata metadata{};
  {
    auto opened = HeadWorker::open(model.string(), 0u);
    ASSERT_TRUE(opened) << opened.error();
    metadata = opened->model_metadata();
  }

  const std::string condition_name = unique_name("flowedge-relay-process-condition");
  const std::string action_name = unique_name("flowedge-relay-process-action");
  auto daemon = ChildDaemon::start(std::filesystem::path{FLOWEDGE_RELAYD_PATH}, model,
                                   condition_name, action_name);
  ASSERT_TRUE(daemon) << "Failed to launch flowedge-relayd";

  std::optional<RelayClient> client{};
  std::string connect_error{};
  const auto connect_timeout = std::chrono::steady_clock::now() + std::chrono::seconds{10};
  do {
    auto connected = RelayClient::connect(condition_name, action_name, metadata);
    if (connected) {
      client.emplace(std::move(*connected));
      break;
    }
    connect_error = connected.error();
    std::this_thread::sleep_for(std::chrono::milliseconds{2});
  } while (std::chrono::steady_clock::now() < connect_timeout);
  ASSERT_TRUE(client) << connect_error;

  const std::vector<float> condition(metadata.condition_dim, 0.125f);
  const std::vector<float> noise(metadata.action_dim, -0.25f);
  const std::uint64_t now = monotonic_ns();
  const RelayRequest request{.sequence = 41u,
                             .session_id = 73u,
                             .timestamp_ns = now,
                             .deadline_ns = now + 5'000'000'000u,
                             .generation = 9u,
                             .solver_steps = 2uz,
                             .solver = FE_SOLVER_HEUN,
                             .condition = condition,
                             .noise = noise};
  ASSERT_EQ(client->try_submit(request), ClientResult::kSuccess);

  ActionMessage action{};
  ClientResult received{ClientResult::kEmpty};
  const auto response_timeout = std::chrono::steady_clock::now() + std::chrono::seconds{10};
  do {
    received = client->try_receive(action);
    if (received != ClientResult::kEmpty)
      break;
    std::this_thread::yield();
  } while (std::chrono::steady_clock::now() < response_timeout);
  ASSERT_EQ(received, ClientResult::kSuccess)
      << to_string(received) << " flags=" << action.envelope.flags
      << " status=" << action.metadata.status << " struct=" << action.envelope.struct_size
      << " wire=" << wire_size(action) << " metadata_struct=" << action.metadata.struct_size
      << " protocol=" << action.metadata.protocol_version;
  EXPECT_EQ(action.envelope.sequence, request.sequence);
  EXPECT_EQ(action.envelope.session_id, request.session_id);
  EXPECT_EQ(action.metadata.generation, request.generation);
  EXPECT_EQ(action.metadata.status, FE_ACTION_COMPLETE);

  const std::uint64_t rejection_time = monotonic_ns();
  const RelayRequest unreachable{.sequence = 42u,
                                 .session_id = request.session_id,
                                 .timestamp_ns = rejection_time,
                                 .deadline_ns = rejection_time + 1'000'000'000u,
                                 .generation = 10u,
                                 .solver_steps = 2uz,
                                 .solver = FE_SOLVER_HEUN,
                                 .condition = condition,
                                 .noise = noise};
  ASSERT_EQ(client->try_submit(unreachable), ClientResult::kSuccess);
  received = ClientResult::kEmpty;
  const auto rejection_timeout = std::chrono::steady_clock::now() + std::chrono::seconds{10};
  do {
    received = client->try_receive(action);
    if (received != ClientResult::kEmpty)
      break;
    std::this_thread::yield();
  } while (std::chrono::steady_clock::now() < rejection_timeout);
  ASSERT_EQ(received, ClientResult::kSuccess) << to_string(received);
  EXPECT_EQ(action.envelope.sequence, unreachable.sequence);
  EXPECT_EQ(action.metadata.status, FE_ACTION_FAILED);
  EXPECT_EQ(action_code(action), RelayActionCode::kRejectedDeadline);

  ClientResult shutdown{ClientResult::kFull};
  const auto shutdown_timeout = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  do {
    shutdown = client->try_shutdown(43u, request.session_id, 0u);
    if (shutdown != ClientResult::kFull)
      break;
    std::this_thread::yield();
  } while (std::chrono::steady_clock::now() < shutdown_timeout);
  ASSERT_EQ(shutdown, ClientResult::kSuccess);

  const auto exit_code = daemon->wait_for(std::chrono::seconds{5});
  ASSERT_TRUE(exit_code) << "flowedge-relayd did not stop after its shutdown control message";
  EXPECT_EQ(*exit_code, 0);
}

TEST(RelayProcess, ExchangesGenericJobWithServiceAcrossProcessBoundary)
{
  const std::string request_name = unique_name("flowedge-job-process-request");
  const std::string result_name = unique_name("flowedge-job-process-result");
  auto daemon =
      ChildDaemon::start_job_service(std::filesystem::path{FLOWEDGE_JOB_SERVICE_FIXTURE_PATH},
                                     request_name, result_name);
  ASSERT_TRUE(daemon) << "Failed to launch generic job service fixture";

  std::optional<JobClient> client{};
  std::string connect_error{};
  const auto connect_timeout = std::chrono::steady_clock::now() + std::chrono::seconds{10};
  do {
    auto connected = JobClient::connect(request_name, result_name);
    if (connected) {
      client.emplace(std::move(*connected));
      break;
    }
    connect_error = connected.error();
    std::this_thread::sleep_for(std::chrono::milliseconds{2});
  } while (std::chrono::steady_clock::now() < connect_timeout);
  ASSERT_TRUE(client) << connect_error;

  JobDescriptor descriptor{};
  descriptor.kind = JobKind::kIterative;
  for (std::size_t index{}; index < descriptor.model_digest.size(); ++index)
    descriptor.model_digest[index] = static_cast<std::uint8_t>(index + 1uz);
  descriptor.state_schema = 0x726f7574652d7631u; // "route-v1"
  descriptor.session_id = 71u;
  descriptor.generation = 3u;
  descriptor.total_work_units = 4u;

  std::array<std::byte, sizeof(std::uint64_t)> input{};
  StateWriter input_writer{input};
  ASSERT_TRUE(input_writer.write(std::uint64_t{5u}));
  auto request = std::make_unique<JobRequestMessage>();
  ASSERT_EQ(make_job_request(*request, 61u, descriptor, monotonic_ns(), input),
            ProtocolResult::kSuccess);
  ASSERT_EQ(client->try_submit(*request), ClientResult::kSuccess);

  auto result = std::make_unique<JobResultMessage>();
  ClientResult received{ClientResult::kEmpty};
  const auto response_timeout = std::chrono::steady_clock::now() + std::chrono::seconds{10};
  do {
    received = client->try_receive(*result);
    if (received != ClientResult::kEmpty)
      break;
    std::this_thread::yield();
  } while (std::chrono::steady_clock::now() < response_timeout);
  ASSERT_EQ(received, ClientResult::kSuccess) << to_string(received);
  EXPECT_EQ(result->envelope.sequence, 61u);
  EXPECT_EQ(job_result_code(*result), JobResultCode::kComplete);
  StateReader result_reader{result->payload_values()};
  const auto value = result_reader.read<std::uint64_t>();
  ASSERT_TRUE(value);
  EXPECT_EQ(*value, 463u);

  ++descriptor.state_schema;
  ++descriptor.generation;
  ASSERT_EQ(make_job_request(*request, 62u, descriptor, monotonic_ns(), input),
            ProtocolResult::kSuccess);
  ASSERT_EQ(client->try_submit(*request), ClientResult::kSuccess);
  received = ClientResult::kEmpty;
  const auto rejection_timeout = std::chrono::steady_clock::now() + std::chrono::seconds{10};
  do {
    received = client->try_receive(*result);
    if (received != ClientResult::kEmpty)
      break;
    std::this_thread::yield();
  } while (std::chrono::steady_clock::now() < rejection_timeout);
  ASSERT_EQ(received, ClientResult::kSuccess) << to_string(received);
  EXPECT_EQ(result->envelope.sequence, 62u);
  EXPECT_EQ(job_result_code(*result), JobResultCode::kAdapterNotFound);

  ASSERT_EQ(client->try_shutdown(63u, descriptor.session_id), ClientResult::kSuccess);
  const auto exit_code = daemon->wait_for(std::chrono::seconds{5});
  ASSERT_TRUE(exit_code) << "generic job service did not stop after shutdown";
  EXPECT_EQ(*exit_code, 0);
}

} // namespace
