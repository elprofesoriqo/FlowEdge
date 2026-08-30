#include "relay/protocol/trace.h"
#include "relay/worker/head_worker.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace {

using fe::relay::ActionMessage;
using fe::relay::ConditionMessage;
using fe::relay::ControlMessage;
using fe::relay::HeadWorker;
using fe::relay::MessageEnvelope;
using fe::relay::MessageKind;
using fe::relay::ProtocolResult;
using fe::relay::TraceReader;
using fe::relay::TraceRecord;
using fe::relay::WorkerStep;

enum class Command : std::uint8_t
{
  kNone,
  kInspect,
  kReplay,
};

struct Options
{
  Command command{Command::kNone};
  std::string trace{};
  std::string model{};
  float tolerance{1.0e-5f};
  std::optional<unsigned> threads{0u};
  bool json_lines{false};
  bool help{false};
};

struct RequestKey
{
  std::uint64_t session_id{};
  std::uint64_t sequence{};
  bool operator==(const RequestKey&) const noexcept = default;
};

struct RequestKeyHash
{
  [[nodiscard]] std::size_t operator()(const RequestKey& key) const noexcept
  {
    const std::uint64_t mixed = key.sequence ^ (key.session_id + 0x9e3779b97f4a7c15u +
                                                (key.sequence << 6u) + (key.sequence >> 2u));
    return static_cast<std::size_t>(mixed);
  }
};

template<typename Number>
[[nodiscard]] bool parse_number(std::string_view text, Number& value) noexcept
{
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc{} && end == text.data() + text.size();
}

void usage(std::ostream& output)
{
  output << "Usage:\n"
            "  flowedge-relay-trace inspect TRACE [--jsonl]\n"
            "  flowedge-relay-trace replay TRACE --model FILE [options]\n"
            "Options:\n"
            "  --threads N        exact Core worker count, 0..8\n"
            "  --tolerance F      absolute action tolerance (default 1e-5)\n";
}

[[nodiscard]] std::expected<Options, std::string> parse_options(int argc, char** argv)
{
  Options options{};
  if (argc < 2)
    return std::unexpected("An inspect or replay command is required");
  const std::string_view command{argv[1]};
  if (command == "--help" || command == "-h") {
    options.help = true;
    return options;
  }
  if (command == "inspect")
    options.command = Command::kInspect;
  else if (command == "replay")
    options.command = Command::kReplay;
  else
    return std::unexpected("Unknown command: " + std::string{command});
  if (argc < 3)
    return std::unexpected("A trace path is required");
  options.trace = argv[2];

  for (int i{3}; i < argc; ++i) {
    const std::string_view argument{argv[i]};
    if (argument == "--help" || argument == "-h") {
      options.help = true;
      return options;
    }
    if (argument == "--jsonl") {
      options.json_lines = true;
      continue;
    }
    if (++i >= argc)
      return std::unexpected("Missing value after " + std::string{argument});
    const std::string_view value{argv[i]};
    if (argument == "--model")
      options.model = value;
    else if (argument == "--threads") {
      unsigned threads{};
      if (!parse_number(value, threads) || threads > 8u)
        return std::unexpected("Invalid --threads value; expected 0..8");
      options.threads = threads;
    } else if (argument == "--tolerance") {
      if (!parse_number(value, options.tolerance) || !std::isfinite(options.tolerance) ||
          options.tolerance < 0.0f)
        return std::unexpected("Invalid --tolerance value");
    } else {
      return std::unexpected("Unknown option: " + std::string{argument});
    }
  }
  if (options.command == Command::kReplay && options.model.empty())
    return std::unexpected("replay requires --model");
  if (options.command != Command::kInspect && options.json_lines)
    return std::unexpected("--jsonl is available only for inspect");
  return options;
}

template<typename Message>
[[nodiscard]] bool unpack(const TraceRecord& record, MessageKind kind, Message& message) noexcept
{
  if (record.kind != kind || record.message.size() > sizeof(Message))
    return false;
  message = {};
  std::memcpy(&message, record.message.data(), record.message.size());
  if constexpr (std::is_same_v<Message, ConditionMessage> || std::is_same_v<Message, ActionMessage>)
    return fe::relay::validate(message) == ProtocolResult::kSuccess &&
           fe::relay::wire_size(message) == record.message.size();
  else
    return record.message.size() == sizeof(ControlMessage) &&
           fe::relay::valid_envelope(message.envelope, MessageKind::kShutdown, sizeof(message));
}

[[nodiscard]] std::string_view action_status(std::uint32_t status) noexcept
{
  switch (status) {
  case FE_ACTION_RUNNING:
    return "running";
  case FE_ACTION_COMPLETE:
    return "complete";
  case FE_ACTION_CANCELLED:
    return "cancelled";
  case FE_ACTION_FAILED:
    return "failed";
  default:
    return "unknown";
  }
}

[[nodiscard]] int inspect_trace(const Options& options)
{
  auto opened = TraceReader::open(options.trace.c_str());
  if (!opened) {
    std::cerr << "flowedge-relay-trace: " << opened.error() << '\n';
    return 1;
  }
  TraceReader reader = std::move(*opened);
  std::uint64_t conditions{};
  std::uint64_t actions{};
  std::uint64_t controls{};
  ConditionMessage condition{};
  ActionMessage action{};
  ControlMessage control{};
  while (true) {
    const auto next = reader.next();
    if (!next) {
      std::cerr << "flowedge-relay-trace: " << next.error() << '\n';
      return 1;
    }
    if (!*next)
      break;
    const TraceRecord& record = **next;
    if (record.kind == MessageKind::kCondition &&
        unpack(record, MessageKind::kCondition, condition)) {
      ++conditions;
      if (options.json_lines) {
        std::cout << "{\"kind\":\"condition\",\"sequence\":" << condition.envelope.sequence
                  << ",\"session\":" << condition.envelope.session_id
                  << ",\"generation\":" << condition.metadata.generation
                  << ",\"timestamp_ns\":" << condition.metadata.timestamp_ns
                  << ",\"deadline_ns\":" << condition.metadata.deadline_ns
                  << ",\"remaining_nfe\":" << condition.metadata.remaining_nfe << "}\n";
      } else {
        std::cout << "condition sequence=" << condition.envelope.sequence
                  << " session=" << condition.envelope.session_id
                  << " generation=" << condition.metadata.generation
                  << " timestamp_ns=" << condition.metadata.timestamp_ns
                  << " deadline_ns=" << condition.metadata.deadline_ns
                  << " remaining_nfe=" << condition.metadata.remaining_nfe << '\n';
      }
    } else if (record.kind == MessageKind::kAction &&
               unpack(record, MessageKind::kAction, action)) {
      ++actions;
      if (options.json_lines) {
        std::cout << "{\"kind\":\"action\",\"sequence\":" << action.envelope.sequence
                  << ",\"session\":" << action.envelope.session_id
                  << ",\"generation\":" << action.metadata.generation << ",\"status\":\""
                  << action_status(action.metadata.status)
                  << "\",\"remaining_nfe\":" << action.metadata.remaining_nfe << "}\n";
      } else {
        std::cout << "action sequence=" << action.envelope.sequence
                  << " session=" << action.envelope.session_id
                  << " generation=" << action.metadata.generation
                  << " status=" << action_status(action.metadata.status)
                  << " remaining_nfe=" << action.metadata.remaining_nfe << '\n';
      }
    } else if (record.kind == MessageKind::kShutdown &&
               unpack(record, MessageKind::kShutdown, control)) {
      ++controls;
      if (options.json_lines)
        std::cout << "{\"kind\":\"shutdown\",\"sequence\":" << control.envelope.sequence
                  << ",\"session\":" << control.envelope.session_id
                  << ",\"reason\":" << control.reason << "}\n";
      else
        std::cout << "shutdown sequence=" << control.envelope.sequence
                  << " session=" << control.envelope.session_id << " reason=" << control.reason
                  << '\n';
    } else {
      std::cerr << "flowedge-relay-trace: decoded record has invalid message semantics\n";
      return 1;
    }
  }
  if (options.json_lines)
    std::cout << "{\"kind\":\"summary\",\"conditions\":" << conditions << ",\"actions\":" << actions
              << ",\"controls\":" << controls << "}\n";
  else
    std::cout << "summary conditions=" << conditions << " actions=" << actions
              << " controls=" << controls << '\n';
  return 0;
}

[[nodiscard]] int replay_trace(const Options& options)
{
  auto worker_result = HeadWorker::open(options.model, options.threads);
  if (!worker_result) {
    std::cerr << "flowedge-relay-trace: " << worker_result.error() << '\n';
    return 1;
  }
  HeadWorker worker = std::move(*worker_result);
  auto opened = TraceReader::open(options.trace.c_str());
  if (!opened) {
    std::cerr << "flowedge-relay-trace: " << opened.error() << '\n';
    return 1;
  }
  TraceReader reader = std::move(*opened);
  std::unordered_map<RequestKey, ActionMessage, RequestKeyHash> replayed{};
  std::uint64_t compared{};
  std::uint64_t mismatched{};
  std::uint64_t skipped{};
  float maximum_error{};
  ConditionMessage condition{};
  ActionMessage recorded{};
  while (true) {
    const auto next = reader.next();
    if (!next) {
      std::cerr << "flowedge-relay-trace: " << next.error() << '\n';
      return 1;
    }
    if (!*next)
      break;
    const TraceRecord& record = **next;
    if (record.kind == MessageKind::kCondition) {
      if (!unpack(record, MessageKind::kCondition, condition) ||
          !fe::relay::compatible(condition, worker.model_metadata())) {
        std::cerr << "flowedge-relay-trace: condition is incompatible with replay model\n";
        return 1;
      }
      if (!worker.begin(condition)) {
        std::cerr << "flowedge-relay-trace: " << worker.last_error() << '\n';
        return 1;
      }
      ActionMessage computed{};
      WorkerStep step{WorkerStep::kInProgress};
      while (step == WorkerStep::kInProgress)
        step = worker.advance(computed);
      if (step != WorkerStep::kComplete) {
        std::cerr << "flowedge-relay-trace: replay worker did not complete request\n";
        return 1;
      }
      replayed.insert_or_assign(RequestKey{condition.envelope.session_id,
                                           condition.envelope.sequence},
                                computed);
    } else if (record.kind == MessageKind::kAction) {
      if (!unpack(record, MessageKind::kAction, recorded) ||
          !fe::relay::compatible(recorded, worker.model_metadata())) {
        std::cerr << "flowedge-relay-trace: action record is invalid\n";
        return 1;
      }
      if (recorded.metadata.status != FE_ACTION_COMPLETE) {
        ++skipped;
        continue;
      }
      const RequestKey key{recorded.envelope.session_id, recorded.envelope.sequence};
      const auto computed = replayed.find(key);
      if (computed == replayed.end()) {
        ++mismatched;
        continue;
      }
      ++compared;
      bool matches = computed->second.metadata.generation == recorded.metadata.generation &&
                     computed->second.metadata.action_dim == recorded.metadata.action_dim;
      for (std::size_t i{0uz}; i < recorded.metadata.action_dim; ++i) {
        const float error = std::abs(computed->second.action[i] - recorded.action[i]);
        maximum_error = std::max(maximum_error, error);
        if (error > options.tolerance)
          matches = false;
      }
      if (!matches)
        ++mismatched;
      replayed.erase(computed);
    }
  }
  std::cout << "replay compared=" << compared << " mismatched=" << mismatched
            << " skipped_noncomplete=" << skipped << " unmatched_conditions=" << replayed.size()
            << " max_abs_error=" << maximum_error << " tolerance=" << options.tolerance << '\n';
  return mismatched == 0u ? 0 : 1;
}

} // namespace

int main(int argc, char** argv)
try {
  const auto parsed = parse_options(argc, argv);
  if (!parsed) {
    std::cerr << "flowedge-relay-trace: " << parsed.error() << '\n';
    usage(std::cerr);
    return 2;
  }
  const Options& options = *parsed;
  if (options.help) {
    usage(std::cout);
    return 0;
  }
  return options.command == Command::kInspect ? inspect_trace(options) : replay_trace(options);
} catch (const std::exception& error) {
  std::cerr << "flowedge-relay-trace: " << error.what() << '\n';
  return 1;
}
