#include "relay/telemetry/metrics.h"

#include <algorithm>
#include <bit>
#include <fstream>
#include <limits>
#include <ostream>
#include <string>
#include <utility>

namespace fe::relay {
namespace {

[[nodiscard]] constexpr std::uint64_t elapsed(std::uint64_t start, std::uint64_t end) noexcept
{
  return end >= start ? end - start : 0u;
}

[[nodiscard]] constexpr std::uint64_t saturating_add(std::uint64_t left,
                                                     std::uint64_t right) noexcept
{
  return left > std::numeric_limits<std::uint64_t>::max() - right
             ? std::numeric_limits<std::uint64_t>::max()
             : left + right;
}

void write_counter(std::ostream& output, std::string_view prefix, std::string_view name,
                   std::uint64_t value)
{
  output << prefix << '_' << name << ' ' << value << '\n';
}

void write_histogram(std::ostream& output, std::string_view prefix, std::string_view name,
                     const LatencyHistogram& histogram)
{
  std::uint64_t cumulative{};
  for (std::size_t index{}; index < histogram.buckets().size(); ++index) {
    cumulative = saturating_add(cumulative, histogram.buckets()[index]);
    output << prefix << '_' << name << "_bucket{le=\"";
    if (index < 64uz)
      output << (std::uint64_t{1u} << index);
    else
      output << "+Inf";
    output << "\"} " << cumulative << '\n';
  }
  output << prefix << '_' << name << "_count " << histogram.count() << '\n';
  output << prefix << '_' << name << "_sum " << histogram.sum_ns() << '\n';
  output << prefix << '_' << name << "_min " << histogram.min_ns() << '\n';
  output << prefix << '_' << name << "_max " << histogram.max_ns() << '\n';
}

void write_otlp_sum(std::ostream& output, std::string_view name, std::uint64_t value, bool& first)
{
  output << (first ? "" : ",") << "{\"name\":\"flowedge.relay." << name
         << "\",\"unit\":\"1\",\"sum\":{\"aggregationTemporality\":"
            "\"AGGREGATION_TEMPORALITY_CUMULATIVE\",\"isMonotonic\":true,"
            "\"dataPoints\":[{\"asInt\":\""
         << value << "\"}]}}";
  first = false;
}

void write_otlp_histogram(std::ostream& output, std::string_view name,
                          const LatencyHistogram& histogram, bool& first)
{
  output << (first ? "" : ",") << "{\"name\":\"flowedge.relay." << name
         << "\",\"unit\":\"ns\",\"histogram\":{\"aggregationTemporality\":"
            "\"AGGREGATION_TEMPORALITY_CUMULATIVE\",\"dataPoints\":[{\"count\":\""
         << histogram.count() << "\",\"sum\":" << histogram.sum_ns()
         << ",\"min\":" << histogram.min_ns() << ",\"max\":" << histogram.max_ns()
         << ",\"bucketCounts\":[";
  for (std::size_t index{}; index < histogram.buckets().size(); ++index)
    output << (index == 0uz ? "\"" : ",\"") << histogram.buckets()[index] << '\"';
  output << "],\"explicitBounds\":[";
  for (std::size_t index{}; index < 64uz; ++index)
    output << (index == 0uz ? "" : ",") << (std::uint64_t{1u} << index);
  output << "]}]}}";
  first = false;
}

[[nodiscard]] constexpr std::size_t job_kind_index(JobKind kind) noexcept
{
  switch (kind) {
  case JobKind::kIterative:
    return 0uz;
  case JobKind::kStreaming:
    return 1uz;
  case JobKind::kSpeculative:
    return 2uz;
  case JobKind::kUnknown:
    break;
  }
  return 3uz;
}

[[nodiscard]] constexpr std::string_view job_kind_name(std::size_t index) noexcept
{
  constexpr std::array names{"iterative", "streaming", "speculative"};
  return index < names.size() ? names[index] : "unknown";
}

void record_job_counters(JobMetricCounters& counters, const JobEventMessage& event) noexcept
{
  ++counters.events;
  const JobEventMetadata& metadata = event.metadata;
  switch (metadata.event) {
  case JobEventKind::kUnknown:
    break;
  case JobEventKind::kAdmitted:
    ++counters.admitted;
    counters.queue_high_watermark =
        std::max(counters.queue_high_watermark, static_cast<std::uint64_t>(metadata.queue_depth));
    break;
  case JobEventKind::kDispatched:
    ++counters.dispatched;
    break;
  case JobEventKind::kStarted:
    ++counters.started;
    break;
  case JobEventKind::kProgress:
    ++counters.progress;
    break;
  case JobEventKind::kPreempted:
    ++counters.preempted;
    break;
  case JobEventKind::kMigrationStarted:
    ++counters.migrations_started;
    break;
  case JobEventKind::kMigrationCompleted:
    ++counters.migrations_completed;
    break;
  case JobEventKind::kCompleted:
    ++counters.completed;
    break;
  case JobEventKind::kCancelled:
    ++counters.cancelled;
    break;
  case JobEventKind::kFailed:
    ++counters.failed;
    break;
  case JobEventKind::kRejected:
    ++counters.rejected;
    switch (metadata.result_code) {
    case JobResultCode::kRejectedStale:
      ++counters.rejected_stale;
      break;
    case JobResultCode::kRejectedDeadline:
      ++counters.rejected_deadline;
      break;
    case JobResultCode::kRejectedCapacity:
      ++counters.rejected_capacity;
      break;
    case JobResultCode::kAdapterNotFound:
      ++counters.adapter_not_found;
      break;
    case JobResultCode::kInvalidRequest:
      ++counters.invalid_request;
      break;
    case JobResultCode::kRejectedQos:
      ++counters.rejected_qos;
      break;
    default:
      break;
    }
    break;
  }
  if (metadata.event == JobEventKind::kCompleted || metadata.event == JobEventKind::kCancelled ||
      metadata.event == JobEventKind::kFailed || metadata.event == JobEventKind::kRejected)
    counters.completed_work_units =
        saturating_add(counters.completed_work_units, metadata.progress.completed_work_units);
}

void write_job_kind_counter(std::ostream& output, std::string_view prefix, std::string_view name,
                            std::string_view kind, std::uint64_t value)
{
  output << prefix << '_' << name << "{kind=\"" << kind << "\"} " << value << '\n';
}

void write_otlp_job_kind_sum(std::ostream& output, std::string_view name, std::string_view kind,
                             std::uint64_t value, bool& first)
{
  output << (first ? "" : ",") << "{\"name\":\"flowedge.relay.job." << name
         << "\",\"unit\":\"1\",\"sum\":{\"aggregationTemporality\":"
            "\"AGGREGATION_TEMPORALITY_CUMULATIVE\",\"isMonotonic\":true,"
            "\"dataPoints\":[{\"attributes\":[{\"key\":\"job.kind\",\"value\":{"
            "\"stringValue\":\""
         << kind << "\"}}],\"asInt\":\"" << value << "\"}]}}";
  first = false;
}

} // namespace

void LatencyHistogram::observe(std::uint64_t nanoseconds) noexcept
{
  const std::size_t bucket =
      nanoseconds <= 1u ? 0uz : static_cast<std::size_t>(std::bit_width(nanoseconds - 1u));
  ++buckets_[std::min(bucket, buckets_.size() - 1uz)];
  sum_ns_ = saturating_add(sum_ns_, nanoseconds);
  min_ns_ = count_ == 0u ? nanoseconds : std::min(min_ns_, nanoseconds);
  max_ns_ = std::max(max_ns_, nanoseconds);
  ++count_;
}

void RelayMetrics::record_condition(bool valid) noexcept
{
  ++counters_.conditions_received;
  if (!valid)
    ++counters_.conditions_corrupt;
}

void RelayMetrics::record_accepted(std::size_t queue_depth) noexcept
{
  ++counters_.conditions_accepted;
  counters_.queue_high_watermark =
      std::max(counters_.queue_high_watermark, static_cast<std::uint64_t>(queue_depth));
}

void RelayMetrics::record_action(const ActionMessage& action, std::uint64_t published_ns,
                                 WorkerTiming timing) noexcept
{
  ++counters_.actions_published;
  switch (action_code(action)) {
  case RelayActionCode::kRejectedStale:
    ++counters_.rejected_stale;
    break;
  case RelayActionCode::kRejectedDeadline:
    ++counters_.rejected_deadline;
    break;
  case RelayActionCode::kRejectedCapacity:
    ++counters_.rejected_capacity;
    break;
  case RelayActionCode::kExpired:
    ++counters_.expired;
    break;
  case RelayActionCode::kInference:
    if (action.metadata.status == FE_ACTION_COMPLETE)
      ++counters_.inference_complete;
    else if (action.metadata.status == FE_ACTION_CANCELLED)
      ++counters_.inference_cancelled;
    else if (action.metadata.status == FE_ACTION_FAILED)
      ++counters_.inference_failed;
    break;
  }
  if (action.metadata.timestamp_ns != 0u)
    end_to_end_latency_.observe(elapsed(action.metadata.timestamp_ns, published_ns));
  if (timing.started_ns != 0u) {
    queue_latency_.observe(elapsed(action.metadata.timestamp_ns, timing.started_ns));
    execution_latency_.observe(elapsed(timing.started_ns, timing.finished_ns));
  }
}

void RelayMetrics::observe_workers(std::size_t busy, std::uint64_t failures) noexcept
{
  counters_.busy_workers_high_watermark =
      std::max(counters_.busy_workers_high_watermark, static_cast<std::uint64_t>(busy));
  counters_.worker_failures = failures;
}

void JobMetrics::record(const JobEventMessage& event) noexcept
{
  if (validate(event) != ProtocolResult::kSuccess)
    return;
  record_job_counters(total_, event);
  const std::size_t index = job_kind_index(event.metadata.descriptor.kind);
  if (index < kinds_.size())
    record_job_counters(kinds_[index], event);

  const bool terminal =
      event.metadata.event == JobEventKind::kCompleted ||
      event.metadata.event == JobEventKind::kCancelled ||
      event.metadata.event == JobEventKind::kFailed ||
      (event.metadata.event == JobEventKind::kRejected && event.metadata.worker_index != kNoWorker);
  if (terminal) {
    queue_latency_.observe(event.metadata.timing.queue_ns);
    execution_latency_.observe(event.metadata.timing.execution_ns);
    end_to_end_latency_.observe(event.metadata.timing.end_to_end_ns);
    if (event.metadata.timing.cancellation_ns != 0u)
      cancellation_latency_.observe(event.metadata.timing.cancellation_ns);
  }
}

void JobMetrics::observe_workers(std::size_t busy, std::uint64_t failures,
                                 std::uint64_t quarantines) noexcept
{
  total_.busy_workers_high_watermark =
      std::max(total_.busy_workers_high_watermark, static_cast<std::uint64_t>(busy));
  total_.worker_failures = failures;
  total_.worker_quarantines = quarantines;
}

const JobMetricCounters* JobMetrics::counters(JobKind kind) const noexcept
{
  const std::size_t index = job_kind_index(kind);
  return index < kinds_.size() ? &kinds_[index] : nullptr;
}

void write_prometheus(std::ostream& output, const RelayMetrics& metrics, std::string_view prefix)
{
  const RelayMetricCounters& counters = metrics.counters();
  write_counter(output, prefix, "conditions_received_total", counters.conditions_received);
  write_counter(output, prefix, "conditions_corrupt_total", counters.conditions_corrupt);
  write_counter(output, prefix, "conditions_accepted_total", counters.conditions_accepted);
  write_counter(output, prefix, "actions_published_total", counters.actions_published);
  write_counter(output, prefix, "actions_dropped_total", counters.actions_dropped);
  write_counter(output, prefix, "inference_complete_total", counters.inference_complete);
  write_counter(output, prefix, "inference_cancelled_total", counters.inference_cancelled);
  write_counter(output, prefix, "inference_failed_total", counters.inference_failed);
  write_counter(output, prefix, "rejected_stale_total", counters.rejected_stale);
  write_counter(output, prefix, "rejected_deadline_total", counters.rejected_deadline);
  write_counter(output, prefix, "rejected_capacity_total", counters.rejected_capacity);
  write_counter(output, prefix, "expired_total", counters.expired);
  write_counter(output, prefix, "queue_high_watermark", counters.queue_high_watermark);
  write_counter(output, prefix, "busy_workers_high_watermark",
                counters.busy_workers_high_watermark);
  write_counter(output, prefix, "worker_failures_total", counters.worker_failures);
  write_histogram(output, prefix, "queue_latency_ns", metrics.queue_latency());
  write_histogram(output, prefix, "execution_latency_ns", metrics.execution_latency());
  write_histogram(output, prefix, "end_to_end_latency_ns", metrics.end_to_end_latency());
}

void write_metrics_json(std::ostream& output, const RelayMetrics& metrics)
{
  const RelayMetricCounters& c = metrics.counters();
  output << "{\"conditions_received\":" << c.conditions_received
         << ",\"conditions_corrupt\":" << c.conditions_corrupt
         << ",\"conditions_accepted\":" << c.conditions_accepted
         << ",\"actions_published\":" << c.actions_published
         << ",\"actions_dropped\":" << c.actions_dropped
         << ",\"inference_complete\":" << c.inference_complete
         << ",\"inference_cancelled\":" << c.inference_cancelled
         << ",\"inference_failed\":" << c.inference_failed
         << ",\"rejected_stale\":" << c.rejected_stale
         << ",\"rejected_deadline\":" << c.rejected_deadline
         << ",\"rejected_capacity\":" << c.rejected_capacity << ",\"expired\":" << c.expired
         << ",\"queue_high_watermark\":" << c.queue_high_watermark
         << ",\"busy_workers_high_watermark\":" << c.busy_workers_high_watermark
         << ",\"worker_failures\":" << c.worker_failures << ",\"latency_ns\":{"
         << "\"queue\":{\"count\":" << metrics.queue_latency().count()
         << ",\"min\":" << metrics.queue_latency().min_ns()
         << ",\"max\":" << metrics.queue_latency().max_ns()
         << ",\"sum\":" << metrics.queue_latency().sum_ns() << "},"
         << "\"execution\":{\"count\":" << metrics.execution_latency().count()
         << ",\"min\":" << metrics.execution_latency().min_ns()
         << ",\"max\":" << metrics.execution_latency().max_ns()
         << ",\"sum\":" << metrics.execution_latency().sum_ns() << "},"
         << "\"end_to_end\":{\"count\":" << metrics.end_to_end_latency().count()
         << ",\"min\":" << metrics.end_to_end_latency().min_ns()
         << ",\"max\":" << metrics.end_to_end_latency().max_ns()
         << ",\"sum\":" << metrics.end_to_end_latency().sum_ns() << "}}}\n";
}

void write_otlp_json(std::ostream& output, const RelayMetrics& metrics,
                     std::string_view service_name)
{
  const RelayMetricCounters& c = metrics.counters();
  output << "{\"resourceMetrics\":[{\"resource\":{\"attributes\":[{\"key\":"
            "\"service.name\",\"value\":{\"stringValue\":\""
         << service_name
         << "\"}}]},\"scopeMetrics\":[{\"scope\":{\"name\":\"flowedge.relay\"},"
            "\"metrics\":[";
  bool first{true};
  const std::array counters{
      std::pair{"conditions_received_total", c.conditions_received},
      std::pair{"conditions_corrupt_total", c.conditions_corrupt},
      std::pair{"conditions_accepted_total", c.conditions_accepted},
      std::pair{"actions_published_total", c.actions_published},
      std::pair{"actions_dropped_total", c.actions_dropped},
      std::pair{"inference_complete_total", c.inference_complete},
      std::pair{"inference_cancelled_total", c.inference_cancelled},
      std::pair{"inference_failed_total", c.inference_failed},
      std::pair{"rejected_stale_total", c.rejected_stale},
      std::pair{"rejected_deadline_total", c.rejected_deadline},
      std::pair{"rejected_capacity_total", c.rejected_capacity},
      std::pair{"expired_total", c.expired},
      std::pair{"queue_high_watermark", c.queue_high_watermark},
      std::pair{"busy_workers_high_watermark", c.busy_workers_high_watermark},
      std::pair{"worker_failures_total", c.worker_failures},
  };
  for (const auto& [name, value] : counters)
    write_otlp_sum(output, name, value, first);
  write_otlp_histogram(output, "queue_latency", metrics.queue_latency(), first);
  write_otlp_histogram(output, "execution_latency", metrics.execution_latency(), first);
  write_otlp_histogram(output, "end_to_end_latency", metrics.end_to_end_latency(), first);
  output << "]}]}]}\n";
}

void write_prometheus(std::ostream& output, const JobMetrics& metrics, std::string_view prefix)
{
  const JobMetricCounters& c = metrics.counters();
  const std::array totals{
      std::pair{"events_total", c.events},
      std::pair{"admitted_total", c.admitted},
      std::pair{"dispatched_total", c.dispatched},
      std::pair{"started_total", c.started},
      std::pair{"progress_total", c.progress},
      std::pair{"completed_total", c.completed},
      std::pair{"cancelled_total", c.cancelled},
      std::pair{"failed_total", c.failed},
      std::pair{"rejected_total", c.rejected},
      std::pair{"rejected_stale_total", c.rejected_stale},
      std::pair{"rejected_deadline_total", c.rejected_deadline},
      std::pair{"rejected_capacity_total", c.rejected_capacity},
      std::pair{"adapter_not_found_total", c.adapter_not_found},
      std::pair{"invalid_request_total", c.invalid_request},
      std::pair{"rejected_qos_total", c.rejected_qos},
      std::pair{"preempted_total", c.preempted},
      std::pair{"migrations_started_total", c.migrations_started},
      std::pair{"migrations_completed_total", c.migrations_completed},
      std::pair{"completed_work_units_total", c.completed_work_units},
      std::pair{"queue_high_watermark", c.queue_high_watermark},
      std::pair{"busy_workers_high_watermark", c.busy_workers_high_watermark},
      std::pair{"worker_failures_total", c.worker_failures},
      std::pair{"worker_quarantines_total", c.worker_quarantines},
      std::pair{"events_dropped_total", c.events_dropped},
  };
  for (const auto& [name, value] : totals)
    write_counter(output, prefix, name, value);
  for (std::size_t index{}; index < 3uz; ++index) {
    const JobMetricCounters* const kind = metrics.counters(static_cast<JobKind>(index + 1uz));
    write_job_kind_counter(output, prefix, "completed_by_kind_total", job_kind_name(index),
                           kind->completed);
    write_job_kind_counter(output, prefix, "cancelled_by_kind_total", job_kind_name(index),
                           kind->cancelled);
    write_job_kind_counter(output, prefix, "failed_by_kind_total", job_kind_name(index),
                           kind->failed);
    write_job_kind_counter(output, prefix, "rejected_by_kind_total", job_kind_name(index),
                           kind->rejected);
    write_job_kind_counter(output, prefix, "completed_work_units_by_kind_total",
                           job_kind_name(index), kind->completed_work_units);
  }
  write_histogram(output, prefix, "queue_latency_ns", metrics.queue_latency());
  write_histogram(output, prefix, "execution_latency_ns", metrics.execution_latency());
  write_histogram(output, prefix, "end_to_end_latency_ns", metrics.end_to_end_latency());
  write_histogram(output, prefix, "cancellation_latency_ns", metrics.cancellation_latency());
}

void write_metrics_json(std::ostream& output, const JobMetrics& metrics)
{
  const JobMetricCounters& c = metrics.counters();
  output << "{\"events\":" << c.events << ",\"admitted\":" << c.admitted
         << ",\"dispatched\":" << c.dispatched << ",\"started\":" << c.started
         << ",\"completed\":" << c.completed << ",\"cancelled\":" << c.cancelled
         << ",\"failed\":" << c.failed << ",\"rejected\":" << c.rejected
         << ",\"rejected_qos\":" << c.rejected_qos << ",\"preempted\":" << c.preempted
         << ",\"migrations_started\":" << c.migrations_started
         << ",\"migrations_completed\":" << c.migrations_completed
         << ",\"completed_work_units\":" << c.completed_work_units
         << ",\"worker_failures\":" << c.worker_failures
         << ",\"worker_quarantines\":" << c.worker_quarantines
         << ",\"events_dropped\":" << c.events_dropped << ",\"kinds\":{";
  for (std::size_t index{}; index < 3uz; ++index) {
    const JobMetricCounters* const kind = metrics.counters(static_cast<JobKind>(index + 1uz));
    output << (index == 0uz ? "" : ",") << '\"' << job_kind_name(index)
           << "\":{\"completed\":" << kind->completed << ",\"cancelled\":" << kind->cancelled
           << ",\"failed\":" << kind->failed << ",\"rejected\":" << kind->rejected
           << ",\"completed_work_units\":" << kind->completed_work_units << '}';
  }
  output << "},\"latency_ns\":{" << "\"queue\":{\"count\":" << metrics.queue_latency().count()
         << ",\"sum\":" << metrics.queue_latency().sum_ns() << "},"
         << "\"execution\":{\"count\":" << metrics.execution_latency().count()
         << ",\"sum\":" << metrics.execution_latency().sum_ns() << "},"
         << "\"end_to_end\":{\"count\":" << metrics.end_to_end_latency().count()
         << ",\"sum\":" << metrics.end_to_end_latency().sum_ns() << "},"
         << "\"cancellation\":{\"count\":" << metrics.cancellation_latency().count()
         << ",\"sum\":" << metrics.cancellation_latency().sum_ns() << "}}}\n";
}

void write_otlp_json(std::ostream& output, const JobMetrics& metrics, std::string_view service_name)
{
  const JobMetricCounters& c = metrics.counters();
  output << "{\"resourceMetrics\":[{\"resource\":{\"attributes\":[{\"key\":"
            "\"service.name\",\"value\":{\"stringValue\":\""
         << service_name
         << "\"}}]},\"scopeMetrics\":[{\"scope\":{\"name\":\"flowedge.relay.jobs\"},"
            "\"metrics\":[";
  bool first{true};
  write_otlp_sum(output, "job.events_total", c.events, first);
  write_otlp_sum(output, "job.admitted_total", c.admitted, first);
  write_otlp_sum(output, "job.completed_total", c.completed, first);
  write_otlp_sum(output, "job.cancelled_total", c.cancelled, first);
  write_otlp_sum(output, "job.failed_total", c.failed, first);
  write_otlp_sum(output, "job.rejected_total", c.rejected, first);
  write_otlp_sum(output, "job.rejected_qos_total", c.rejected_qos, first);
  write_otlp_sum(output, "job.preempted_total", c.preempted, first);
  write_otlp_sum(output, "job.migrations_completed_total", c.migrations_completed, first);
  write_otlp_sum(output, "job.completed_work_units_total", c.completed_work_units, first);
  write_otlp_sum(output, "job.worker_failures_total", c.worker_failures, first);
  write_otlp_sum(output, "job.worker_quarantines_total", c.worker_quarantines, first);
  for (std::size_t index{}; index < 3uz; ++index) {
    const JobMetricCounters* const kind = metrics.counters(static_cast<JobKind>(index + 1uz));
    write_otlp_job_kind_sum(output, "completed_by_kind_total", job_kind_name(index),
                            kind->completed, first);
    write_otlp_job_kind_sum(output, "completed_work_units_by_kind_total", job_kind_name(index),
                            kind->completed_work_units, first);
  }
  write_otlp_histogram(output, "job.queue_latency", metrics.queue_latency(), first);
  write_otlp_histogram(output, "job.execution_latency", metrics.execution_latency(), first);
  write_otlp_histogram(output, "job.end_to_end_latency", metrics.end_to_end_latency(), first);
  write_otlp_histogram(output, "job.cancellation_latency", metrics.cancellation_latency(), first);
  output << "]}]}]}\n";
}

std::expected<void, std::string> export_prometheus(std::string_view path,
                                                   const RelayMetrics& metrics) noexcept
{
  try {
    std::ofstream output{std::string{path}, std::ios::trunc};
    if (!output)
      return std::unexpected("failed to open Prometheus metrics file");
    write_prometheus(output, metrics);
    if (!output)
      return std::unexpected("failed to write Prometheus metrics file");
    return {};
  } catch (...) {
    return std::unexpected("exception exporting Prometheus metrics");
  }
}

std::expected<void, std::string> export_metrics_json(std::string_view path,
                                                     const RelayMetrics& metrics) noexcept
{
  try {
    std::ofstream output{std::string{path}, std::ios::trunc};
    if (!output)
      return std::unexpected("failed to open JSON metrics file");
    write_metrics_json(output, metrics);
    if (!output)
      return std::unexpected("failed to write JSON metrics file");
    return {};
  } catch (...) {
    return std::unexpected("exception exporting JSON metrics");
  }
}

std::expected<void, std::string> export_otlp_json(std::string_view path,
                                                  const RelayMetrics& metrics,
                                                  std::string_view service_name) noexcept
{
  try {
    std::ofstream output{std::string{path}, std::ios::trunc};
    if (!output)
      return std::unexpected("failed to open OTLP JSON metrics file");
    write_otlp_json(output, metrics, service_name);
    if (!output)
      return std::unexpected("failed to write OTLP JSON metrics file");
    return {};
  } catch (...) {
    return std::unexpected("exception exporting OTLP JSON metrics");
  }
}

std::expected<void, std::string> export_prometheus(std::string_view path,
                                                   const JobMetrics& metrics) noexcept
{
  try {
    std::ofstream output{std::string{path}, std::ios::trunc};
    if (!output)
      return std::unexpected("failed to open generic job Prometheus metrics file");
    write_prometheus(output, metrics);
    return output ? std::expected<void, std::string>{}
                  : std::unexpected("failed to write generic job Prometheus metrics file");
  } catch (...) {
    return std::unexpected("exception exporting generic job Prometheus metrics");
  }
}

std::expected<void, std::string> export_metrics_json(std::string_view path,
                                                     const JobMetrics& metrics) noexcept
{
  try {
    std::ofstream output{std::string{path}, std::ios::trunc};
    if (!output)
      return std::unexpected("failed to open generic job JSON metrics file");
    write_metrics_json(output, metrics);
    return output ? std::expected<void, std::string>{}
                  : std::unexpected("failed to write generic job JSON metrics file");
  } catch (...) {
    return std::unexpected("exception exporting generic job JSON metrics");
  }
}

std::expected<void, std::string> export_otlp_json(std::string_view path, const JobMetrics& metrics,
                                                  std::string_view service_name) noexcept
{
  try {
    std::ofstream output{std::string{path}, std::ios::trunc};
    if (!output)
      return std::unexpected("failed to open generic job OTLP JSON metrics file");
    write_otlp_json(output, metrics, service_name);
    return output ? std::expected<void, std::string>{}
                  : std::unexpected("failed to write generic job OTLP JSON metrics file");
  } catch (...) {
    return std::unexpected("exception exporting generic job OTLP JSON metrics");
  }
}

} // namespace fe::relay
