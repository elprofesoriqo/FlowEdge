#include "api/engine.h"
#include "relay/adapters/mamba_stream_adapter.h"
#include "relay/protocol/trace.h"
#include "relay/telemetry/job_event_buffer.h"
#include "relay/telemetry/metrics.h"
#include "relay/worker/job_control_service.h"
#include "relay/worker/job_service.h"
#include "relay/worker/job_worker_pool.h"
#include "relay/worker/worker_topology.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <expected>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using namespace fe::relay;

struct Options
{
  std::string model{};
  std::string request_shm{"flowedge-job-requests"};
  std::string result_shm{"flowedge-job-results"};
  std::string control_shm{"flowedge-job-control"};
  std::string status_shm{"flowedge-job-status"};
  std::string trace{};
  std::string metrics_prometheus{};
  std::string metrics_json{};
  std::string metrics_otlp_json{};
  std::uint32_t capacity{32u};
  std::uint32_t control_capacity{8u};
  std::size_t workers{1uz};
  std::size_t max_tokens{512uz};
  std::size_t max_sessions{64uz};
  std::size_t work_quantum{1uz};
  std::size_t event_capacity{1'024uz};
  std::size_t interactive_reserve{2uz};
  std::size_t critical_reserve{1uz};
  std::size_t failure_threshold{3uz};
  std::uint64_t streaming_unit_ns{};
  std::uint64_t admission_reserve_ns{};
  std::uint64_t metrics_interval_ms{};
  std::optional<unsigned> threads{0u};
  WorkerPlacement placement{WorkerPlacement::kNone};
  bool create{};
  bool help{};
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic_bool stop_requested{false};

void request_stop(int) noexcept
{
  stop_requested.store(true, std::memory_order_relaxed);
}

template<typename Integer>
[[nodiscard]] bool parse_integer(std::string_view text, Integer& value) noexcept
{
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc{} && end == text.data() + text.size();
}

void usage(std::ostream& output)
{
  output << "Usage: flowedge-jobd --model FILE [options]\n"
            "  --request-shm NAME       job request ring\n"
            "  --result-shm NAME        job result ring\n"
            "  --control-shm NAME       administration request ring\n"
            "  --status-shm NAME        administration response ring\n"
            "  --capacity N             data ring and queue capacity (default 32)\n"
            "  --control-capacity N     control ring capacity (default 8)\n"
            "  --workers N              Mamba execution lanes, 1..8\n"
            "  --threads N              Core threads per lane, 0..8\n"
            "  --placement POLICY       none, compact, or spread\n"
            "  --max-tokens N           tokens accepted per streaming job\n"
            "  --max-sessions N         retained freshness watermarks\n"
            "  --work-quantum N         tokens per cancellation boundary\n"
            "  --streaming-unit-ns N    calibrated deadline cost per token\n"
            "  --admission-reserve-ns N fixed deadline reserve\n"
            "  --event-capacity N       bounded lifecycle event count\n"
            "  --interactive-reserve N queue slots unavailable to best-effort jobs\n"
            "  --critical-reserve N    queue slots reserved for critical jobs\n"
            "  --failure-threshold N   consecutive failures before lane quarantine; 0 disables\n"
            "  --trace FILE             write lifecycle event trace\n"
            "  --metrics-prometheus FILE\n"
            "  --metrics-json FILE\n"
            "  --metrics-otlp-json FILE\n"
            "  --metrics-interval-ms N  0 exports only at shutdown\n"
            "  --create                  create all four rings\n";
}

[[nodiscard]] std::expected<Options, std::string> parse_options(int argc, char** argv)
{
  Options options{};
  for (int index{1}; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "--help" || argument == "-h") {
      options.help = true;
      return options;
    }
    if (argument == "--create") {
      options.create = true;
      continue;
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
    else if (argument == "--trace")
      options.trace = value;
    else if (argument == "--metrics-prometheus")
      options.metrics_prometheus = value;
    else if (argument == "--metrics-json")
      options.metrics_json = value;
    else if (argument == "--metrics-otlp-json")
      options.metrics_otlp_json = value;
    else if (argument == "--capacity") {
      if (!parse_integer(value, options.capacity))
        return std::unexpected("Invalid --capacity value");
    } else if (argument == "--control-capacity") {
      if (!parse_integer(value, options.control_capacity))
        return std::unexpected("Invalid --control-capacity value");
    } else if (argument == "--workers") {
      if (!parse_integer(value, options.workers) || options.workers == 0uz || options.workers > 8uz)
        return std::unexpected("Invalid --workers value; expected 1..8");
    } else if (argument == "--threads") {
      unsigned threads{};
      if (!parse_integer(value, threads) || threads > 8u)
        return std::unexpected("Invalid --threads value; expected 0..8");
      options.threads = threads;
    } else if (argument == "--placement") {
      const auto placement = parse_worker_placement(value);
      if (!placement)
        return std::unexpected(std::string{placement.error()});
      options.placement = *placement;
    } else if (argument == "--max-tokens") {
      if (!parse_integer(value, options.max_tokens) || options.max_tokens == 0uz)
        return std::unexpected("Invalid --max-tokens value");
    } else if (argument == "--max-sessions") {
      if (!parse_integer(value, options.max_sessions) || options.max_sessions == 0uz)
        return std::unexpected("Invalid --max-sessions value");
    } else if (argument == "--work-quantum") {
      if (!parse_integer(value, options.work_quantum) || options.work_quantum == 0uz)
        return std::unexpected("Invalid --work-quantum value");
    } else if (argument == "--event-capacity") {
      if (!parse_integer(value, options.event_capacity) || options.event_capacity == 0uz)
        return std::unexpected("Invalid --event-capacity value");
    } else if (argument == "--interactive-reserve") {
      if (!parse_integer(value, options.interactive_reserve))
        return std::unexpected("Invalid --interactive-reserve value");
    } else if (argument == "--critical-reserve") {
      if (!parse_integer(value, options.critical_reserve))
        return std::unexpected("Invalid --critical-reserve value");
    } else if (argument == "--failure-threshold") {
      if (!parse_integer(value, options.failure_threshold))
        return std::unexpected("Invalid --failure-threshold value");
    } else if (argument == "--streaming-unit-ns") {
      if (!parse_integer(value, options.streaming_unit_ns))
        return std::unexpected("Invalid --streaming-unit-ns value");
    } else if (argument == "--admission-reserve-ns") {
      if (!parse_integer(value, options.admission_reserve_ns))
        return std::unexpected("Invalid --admission-reserve-ns value");
    } else if (argument == "--metrics-interval-ms") {
      if (!parse_integer(value, options.metrics_interval_ms) ||
          options.metrics_interval_ms > 86'400'000u)
        return std::unexpected("Invalid --metrics-interval-ms value; maximum is one day");
    } else {
      return std::unexpected("Unknown option: " + std::string{argument});
    }
  }
  if (options.model.empty())
    return std::unexpected("--model is required");
  if (options.request_shm.empty() || options.result_shm.empty() || options.control_shm.empty() ||
      options.status_shm.empty())
    return std::unexpected("Shared-memory ring names cannot be empty");
  if (mamba_stream_request_bytes(options.max_tokens) == 0uz)
    return std::unexpected("--max-tokens exceeds the job payload capacity");
  if (options.interactive_reserve >= options.capacity ||
      options.critical_reserve >= options.capacity ||
      options.interactive_reserve > options.capacity - 1u - options.critical_reserve)
    return std::unexpected("QoS reserves must leave one best-effort queue slot");
  return options;
}

[[nodiscard]] bool export_metrics(const Options& options, const JobMetrics& metrics)
{
  if (!options.metrics_prometheus.empty()) {
    const auto exported = fe::relay::export_prometheus(options.metrics_prometheus, metrics);
    if (!exported) {
      std::cerr << "flowedge-jobd: " << exported.error() << '\n';
      return false;
    }
  }
  if (!options.metrics_json.empty()) {
    const auto exported = fe::relay::export_metrics_json(options.metrics_json, metrics);
    if (!exported) {
      std::cerr << "flowedge-jobd: " << exported.error() << '\n';
      return false;
    }
  }
  if (!options.metrics_otlp_json.empty()) {
    const auto exported =
        fe::relay::export_otlp_json(options.metrics_otlp_json, metrics, "flowedge-jobd");
    if (!exported) {
      std::cerr << "flowedge-jobd: " << exported.error() << '\n';
      return false;
    }
  }
  return true;
}

} // namespace

int main(int argc, char** argv)
try {
  const auto parsed = parse_options(argc, argv);
  if (!parsed) {
    std::cerr << "flowedge-jobd: " << parsed.error() << '\n';
    usage(std::cerr);
    return 2;
  }
  const Options& options = *parsed;
  if (options.help) {
    usage(std::cout);
    return 0;
  }

  std::unique_ptr<fe_weights, decltype(&fe_weights_free)> weights{fe_weights_load(
                                                                      options.model.c_str()),
                                                                  fe_weights_free};
  if (!weights) {
    std::cerr << "flowedge-jobd: " << fe_engine_last_error() << '\n';
    return 1;
  }

  std::vector<MambaStreamAdapter> adapters{};
  std::array<JobAdapterRegistration, 8uz> registrations{};
  std::vector<JobAdapterRegistry> lanes{};
  adapters.reserve(options.workers);
  lanes.reserve(options.workers);
  std::size_t migration_capacity{};
  for (std::size_t index{}; index < options.workers; ++index) {
    auto opened = MambaStreamAdapter::open(weights.get(), options.max_tokens, options.threads);
    if (!opened) {
      std::cerr << "flowedge-jobd: " << opened.error() << '\n';
      return 1;
    }
    adapters.push_back(std::move(*opened));
    lanes.emplace_back(std::span{registrations}.subspan(index, 1uz));
    const auto added = lanes.back().add(adapters.back().registration());
    if (!added) {
      std::cerr << "flowedge-jobd: " << added.error().message << '\n';
      return 1;
    }
    lanes.back().freeze();
    migration_capacity = std::max(migration_capacity, adapters.back().max_state_bytes());
  }
  const JobRoute route = adapters.front().route();
  const std::size_t shared_weight_bytes = fe_weights_size_bytes(weights.get());
  weights.reset();

  auto events_result = JobEventBuffer::create(options.event_capacity);
  if (!events_result) {
    std::cerr << "flowedge-jobd: " << events_result.error() << '\n';
    return 1;
  }
  JobEventBuffer events = std::move(*events_result);
  auto pool_result =
      JobWorkerPool::create(lanes, options.capacity,
                            JobCostPolicy{.streaming_ns = options.streaming_unit_ns,
                                          .reserve_ns = options.admission_reserve_ns},
                            options.max_sessions, options.work_quantum, options.placement, &events,
                            migration_capacity,
                            JobQosPolicy{.interactive_reserve_slots = options.interactive_reserve,
                                         .critical_reserve_slots = options.critical_reserve},
                            WorkerSupervisionPolicy{.consecutive_failure_threshold =
                                                        options.failure_threshold});
  if (!pool_result) {
    std::cerr << "flowedge-jobd: " << pool_result.error() << '\n';
    return 1;
  }
  JobWorkerPool pool = std::move(*pool_result);

  auto service_result =
      options.create
          ? JobService::create(options.request_shm, options.result_shm, pool, options.capacity)
          : JobService::connect(options.request_shm, options.result_shm, pool);
  if (!service_result) {
    std::cerr << "flowedge-jobd: " << service_result.error() << '\n';
    return 1;
  }
  JobService service = std::move(*service_result);
  auto control_result =
      options.create ? JobControlService::create(options.control_shm, options.status_shm, pool,
                                                 service.stats(), options.control_capacity)
                     : JobControlService::connect(options.control_shm, options.status_shm, pool,
                                                  service.stats());
  if (!control_result) {
    std::cerr << "flowedge-jobd: " << control_result.error() << '\n';
    return 1;
  }
  JobControlService control = std::move(*control_result);

  std::optional<TraceWriter> trace{};
  if (!options.trace.empty()) {
    auto opened = TraceWriter::open(options.trace.c_str());
    if (!opened) {
      std::cerr << "flowedge-jobd: " << opened.error() << '\n';
      return 1;
    }
    trace.emplace(std::move(*opened));
  }

  std::signal(SIGINT, request_stop);
  std::signal(SIGTERM, request_stop);
  JobMetrics metrics{};
  auto next_metrics =
      std::chrono::steady_clock::now() + std::chrono::milliseconds{options.metrics_interval_ms};
  bool failed{};
  while (!stop_requested.load(std::memory_order_relaxed) && !service.stopped() &&
         !control.stopped()) {
    const JobServiceResult data_result = service.poll();
    const JobControlServiceResult control_poll = control.poll();
    if (data_result == JobServiceResult::kCorruptInput ||
        data_result == JobServiceResult::kTransportError) {
      std::cerr << "flowedge-jobd: data plane " << to_string(data_result) << '\n';
      failed = true;
      break;
    }
    if (control_poll == JobControlServiceResult::kCorruptInput ||
        control_poll == JobControlServiceResult::kTransportError) {
      std::cerr << "flowedge-jobd: control plane " << to_string(control_poll) << '\n';
      failed = true;
      break;
    }

    while (const JobEventMessage* event = events.front()) {
      metrics.record(*event);
      if (trace) {
        const auto written = trace->append(*event);
        if (!written) {
          std::cerr << "flowedge-jobd: " << written.error() << '\n';
          failed = true;
          break;
        }
      }
      events.pop();
    }
    if (failed)
      break;
    metrics.observe_workers(pool.busy_count(), pool.failure_count(), pool.quarantine_count());
    metrics.record_event_drops(events.dropped());
    if (options.metrics_interval_ms != 0u && std::chrono::steady_clock::now() >= next_metrics) {
      if (!export_metrics(options, metrics)) {
        failed = true;
        break;
      }
      next_metrics =
          std::chrono::steady_clock::now() + std::chrono::milliseconds{options.metrics_interval_ms};
    }
    if (data_result == JobServiceResult::kIdle && control_poll == JobControlServiceResult::kIdle)
      std::this_thread::yield();
  }

  while (const JobEventMessage* event = events.front()) {
    metrics.record(*event);
    if (trace) {
      const auto written = trace->append(*event);
      if (!written) {
        std::cerr << "flowedge-jobd: " << written.error() << '\n';
        failed = true;
      }
    }
    events.pop();
  }
  metrics.observe_workers(pool.busy_count(), pool.failure_count(), pool.quarantine_count());
  metrics.record_event_drops(events.dropped());
  if (trace)
    trace->flush();
  if (!export_metrics(options, metrics))
    failed = true;

  const JobTransportStats& stats = service.stats();
  std::cout << "flowedge-jobd stopped: route=" << to_string(route.kind)
            << " workers=" << pool.worker_count() << " shared_weight_bytes=" << shared_weight_bytes
            << " accepted=" << stats.requests_accepted << " rejected=" << stats.requests_rejected
            << " published=" << stats.results_published
            << " worker_failures=" << pool.failure_count() << " event_drops=" << events.dropped()
            << " worker_quarantines=" << pool.quarantine_count()
            << " qos_rejections=" << pool.qos_rejection_count() << '\n';
  return failed ? 1 : 0;
} catch (const std::exception& error) {
  std::cerr << "flowedge-jobd: " << error.what() << '\n';
  return 1;
} catch (...) {
  std::cerr << "flowedge-jobd: unknown failure\n";
  return 1;
}
