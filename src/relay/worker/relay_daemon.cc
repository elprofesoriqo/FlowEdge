#include "relay/protocol/trace.h"
#include "relay/scheduler/edf_scheduler.h"
#include "relay/shared_memory/shared_memory_ring.h"
#include "relay/worker/head_worker.h"

#include <atomic>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace {

using fe::relay::ActionMessage;
using fe::relay::ConditionMessage;
using fe::relay::ControlMessage;
using fe::relay::EdfScheduler;
using fe::relay::HeadWorker;
using fe::relay::MessageEnvelope;
using fe::relay::MessageKind;
using fe::relay::RingConfig;
using fe::relay::RingResult;
using fe::relay::SharedMemoryRing;
using fe::relay::SubmitResult;
using fe::relay::TraceWriter;
using fe::relay::WorkerStep;

struct Options
{
  std::string model{};
  std::string condition_shm{"flowedge-relay-conditions"};
  std::string action_shm{"flowedge-relay-actions"};
  std::string trace{};
  std::uint32_t capacity{32u};
  std::optional<unsigned> threads{};
  bool create{false};
  bool help{false};
};

// The signal handler and event loop intentionally share this lock-free stop flag.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic_bool stop_requested{false};

void request_stop(int) noexcept
{
  stop_requested.store(true, std::memory_order_relaxed);
}

[[nodiscard]] std::uint64_t monotonic_ns() noexcept
{
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now().time_since_epoch())
                                        .count());
}

template<typename Integer>
[[nodiscard]] bool parse_integer(std::string_view text, Integer& value) noexcept
{
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc{} && end == text.data() + text.size();
}

void usage(std::ostream& output)
{
  output << "Usage: flowedge-relayd --model FILE [options]\n"
            "  --condition-shm NAME  input shared-memory ring name\n"
            "  --action-shm NAME     output shared-memory ring name\n"
            "  --capacity N          power-of-two ring/scheduler capacity (default 32)\n"
            "  --threads N           exact Core worker count, 0..8\n"
            "  --trace FILE          record accepted conditions and emitted actions\n"
            "  --create              create both rings instead of opening them\n";
}

[[nodiscard]] std::expected<Options, std::string> parse_options(int argc, char** argv)
{
  Options options{};
  for (int i{1}; i < argc; ++i) {
    const std::string_view argument{argv[i]};
    auto next = [&]() -> std::expected<std::string_view, std::string> {
      if (++i >= argc)
        return std::unexpected("Missing value after " + std::string{argument});
      return std::string_view{argv[i]};
    };
    if (argument == "--help" || argument == "-h") {
      options.help = true;
      return options;
    }
    if (argument == "--create") {
      options.create = true;
      continue;
    }
    const auto value = next();
    if (!value)
      return std::unexpected(value.error());
    if (argument == "--model")
      options.model = *value;
    else if (argument == "--condition-shm")
      options.condition_shm = *value;
    else if (argument == "--action-shm")
      options.action_shm = *value;
    else if (argument == "--trace")
      options.trace = *value;
    else if (argument == "--capacity") {
      if (!parse_integer(*value, options.capacity))
        return std::unexpected("Invalid --capacity value");
    } else if (argument == "--threads") {
      unsigned threads{};
      if (!parse_integer(*value, threads) || threads > 8u)
        return std::unexpected("Invalid --threads value; expected 0..8");
      options.threads = threads;
    } else {
      return std::unexpected("Unknown option: " + std::string{argument});
    }
  }
  if (options.help)
    return options;
  if (options.model.empty())
    return std::unexpected("--model is required");
  if (options.condition_shm.empty() || options.action_shm.empty())
    return std::unexpected("Shared-memory ring names cannot be empty");
  return options;
}

[[nodiscard]] bool accepted(SubmitResult result) noexcept
{
  return result == SubmitResult::kAccepted || result == SubmitResult::kAcceptedAndEvicted;
}

} // namespace

int main(int argc, char** argv)
try {
  const auto parsed = parse_options(argc, argv);
  if (!parsed) {
    std::cerr << "flowedge-relayd: " << parsed.error() << '\n';
    usage(std::cerr);
    return 2;
  }
  const Options& options = *parsed;
  if (options.help) {
    usage(std::cout);
    return 0;
  }

  auto conditions_result =
      options.create ? SharedMemoryRing::create(options.condition_shm,
                                                RingConfig{.capacity = options.capacity,
                                                           .slot_bytes = sizeof(ConditionMessage)})
                     : SharedMemoryRing::open(options.condition_shm);
  if (!conditions_result) {
    std::cerr << "flowedge-relayd: " << conditions_result.error() << '\n';
    return 1;
  }
  auto actions_result =
      options.create ? SharedMemoryRing::create(options.action_shm,
                                                RingConfig{.capacity = options.capacity,
                                                           .slot_bytes = sizeof(ActionMessage)})
                     : SharedMemoryRing::open(options.action_shm);
  if (!actions_result) {
    std::cerr << "flowedge-relayd: " << actions_result.error() << '\n';
    return 1;
  }
  SharedMemoryRing conditions = std::move(*conditions_result);
  SharedMemoryRing actions = std::move(*actions_result);

  auto worker_result = HeadWorker::open(options.model, options.threads);
  if (!worker_result) {
    std::cerr << "flowedge-relayd: " << worker_result.error() << '\n';
    return 1;
  }
  HeadWorker worker = std::move(*worker_result);
  EdfScheduler scheduler{options.capacity};

  std::optional<TraceWriter> trace{};
  if (!options.trace.empty()) {
    auto trace_result = TraceWriter::open(options.trace.c_str());
    if (!trace_result) {
      std::cerr << "flowedge-relayd: " << trace_result.error() << '\n';
      return 1;
    }
    trace.emplace(std::move(*trace_result));
  }

  std::signal(SIGINT, request_stop);
  std::signal(SIGTERM, request_stop);

  auto incoming = std::make_unique<ConditionMessage>();
  ActionMessage action{};
  bool action_pending{false};
  while (!stop_requested.load(std::memory_order_relaxed)) {
    bool progressed{false};

    const std::uint32_t drain_budget = conditions.config().capacity;
    for (std::uint32_t drained{0u}; drained < drain_budget; ++drained) {
      std::size_t bytes_read{};
      const RingResult result =
          conditions.try_pop(std::as_writable_bytes(std::span{incoming.get(), 1uz}), bytes_read);
      if (result == RingResult::kEmpty)
        break;
      if (result != RingResult::kSuccess) {
        std::cerr << "flowedge-relayd: discarded corrupt condition-ring entry\n";
        progressed = true;
        continue;
      }
      progressed = true;
      if (bytes_read < sizeof(MessageEnvelope))
        continue;
      const MessageEnvelope envelope = incoming->envelope;
      if (envelope.kind == MessageKind::kShutdown && bytes_read == sizeof(ControlMessage) &&
          fe::relay::valid_envelope(envelope, MessageKind::kShutdown, sizeof(ControlMessage))) {
        stop_requested.store(true, std::memory_order_relaxed);
        break;
      }
      if (bytes_read != wire_size(*incoming))
        continue;
      if (!fe::relay::compatible(*incoming, worker.model_metadata()))
        continue;
      const SubmitResult submitted = scheduler.submit(*incoming);
      if (accepted(submitted)) {
        worker.cancel_before(scheduler.newest_generation());
        if (action_pending && action.metadata.generation < scheduler.newest_generation())
          action_pending = false;
        if (trace) {
          const auto written = trace->append(*incoming);
          if (!written)
            std::cerr << "flowedge-relayd: " << written.error() << '\n';
        }
      }
    }

    if (action_pending) {
      const RingResult pushed = actions.try_push(wire_bytes(action));
      if (pushed == RingResult::kSuccess) {
        if (trace) {
          const auto written = trace->append(action);
          if (!written)
            std::cerr << "flowedge-relayd: " << written.error() << '\n';
        }
        action_pending = false;
        progressed = true;
      } else if (pushed != RingResult::kFull) {
        std::cerr << "flowedge-relayd: action message exceeds output ring slot\n";
        return 1;
      }
    }

    if (!action_pending && !worker.busy()) {
      if (const ConditionMessage* request = scheduler.pop(monotonic_ns())) {
        if (!worker.begin(*request))
          std::cerr << "flowedge-relayd: rejected scheduled request: " << worker.last_error()
                    << '\n';
        progressed = true;
      }
    }

    if (!action_pending && worker.busy()) {
      const WorkerStep step = worker.advance(action);
      if (step == WorkerStep::kComplete || step == WorkerStep::kCancelled) {
        action_pending = true;
      } else if (step == WorkerStep::kError) {
        std::cerr << "flowedge-relayd: worker error: " << worker.last_error() << '\n';
      }
      progressed = true;
    }

    if (!progressed)
      std::this_thread::yield();
  }

  if (trace)
    trace->flush();
  const auto& stats = scheduler.stats();
  std::cout << "flowedge-relayd stopped: accepted=" << stats.accepted << " stale=" << stats.stale
            << " expired=" << stats.expired << " evicted=" << stats.evicted
            << " full=" << stats.full << '\n';
  return 0;
} catch (const std::exception& error) {
  std::cerr << "flowedge-relayd: " << error.what() << '\n';
  return 1;
}
