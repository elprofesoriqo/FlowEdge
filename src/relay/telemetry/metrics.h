#pragma once

#include "relay/protocol/job_events.h"
#include "relay/protocol/messages.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <iosfwd>
#include <string>
#include <string_view>

namespace fe::relay {

struct WorkerTiming
{
  std::uint64_t dispatched_ns{};
  std::uint64_t started_ns{};
  std::uint64_t finished_ns{};
};

class LatencyHistogram
{
public:
  static constexpr std::size_t kBuckets = 65uz;

  void observe(std::uint64_t nanoseconds) noexcept;

  [[nodiscard]] std::uint64_t count() const noexcept { return count_; }
  [[nodiscard]] std::uint64_t sum_ns() const noexcept { return sum_ns_; }
  [[nodiscard]] std::uint64_t min_ns() const noexcept { return min_ns_; }
  [[nodiscard]] std::uint64_t max_ns() const noexcept { return max_ns_; }
  [[nodiscard]] const std::array<std::uint64_t, kBuckets>& buckets() const noexcept
  {
    return buckets_;
  }

private:
  std::array<std::uint64_t, kBuckets> buckets_{};
  std::uint64_t count_{};
  std::uint64_t sum_ns_{};
  std::uint64_t min_ns_{};
  std::uint64_t max_ns_{};
};

struct RelayMetricCounters
{
  std::uint64_t conditions_received{};
  std::uint64_t conditions_corrupt{};
  std::uint64_t conditions_accepted{};
  std::uint64_t actions_published{};
  std::uint64_t actions_dropped{};
  std::uint64_t inference_complete{};
  std::uint64_t inference_cancelled{};
  std::uint64_t inference_failed{};
  std::uint64_t rejected_stale{};
  std::uint64_t rejected_deadline{};
  std::uint64_t rejected_capacity{};
  std::uint64_t expired{};
  std::uint64_t queue_high_watermark{};
  std::uint64_t busy_workers_high_watermark{};
  std::uint64_t worker_failures{};
};

class RelayMetrics
{
public:
  void record_condition(bool valid) noexcept;
  void record_accepted(std::size_t queue_depth) noexcept;
  void record_action(const ActionMessage& action, std::uint64_t published_ns,
                     WorkerTiming timing = {}) noexcept;
  void record_action_drop() noexcept { ++counters_.actions_dropped; }
  void observe_workers(std::size_t busy, std::uint64_t failures) noexcept;

  [[nodiscard]] const RelayMetricCounters& counters() const noexcept { return counters_; }
  [[nodiscard]] const LatencyHistogram& queue_latency() const noexcept { return queue_latency_; }
  [[nodiscard]] const LatencyHistogram& execution_latency() const noexcept
  {
    return execution_latency_;
  }
  [[nodiscard]] const LatencyHistogram& end_to_end_latency() const noexcept
  {
    return end_to_end_latency_;
  }

private:
  RelayMetricCounters counters_{};
  LatencyHistogram queue_latency_{};
  LatencyHistogram execution_latency_{};
  LatencyHistogram end_to_end_latency_{};
};

struct JobMetricCounters
{
  std::uint64_t events{};
  std::uint64_t admitted{};
  std::uint64_t dispatched{};
  std::uint64_t started{};
  std::uint64_t progress{};
  std::uint64_t completed{};
  std::uint64_t cancelled{};
  std::uint64_t failed{};
  std::uint64_t rejected{};
  std::uint64_t rejected_stale{};
  std::uint64_t rejected_deadline{};
  std::uint64_t rejected_capacity{};
  std::uint64_t adapter_not_found{};
  std::uint64_t invalid_request{};
  std::uint64_t preempted{};
  std::uint64_t migrations_started{};
  std::uint64_t migrations_completed{};
  std::uint64_t completed_work_units{};
  std::uint64_t queue_high_watermark{};
  std::uint64_t busy_workers_high_watermark{};
  std::uint64_t worker_failures{};
  std::uint64_t events_dropped{};
};

class JobMetrics
{
public:
  void record(const JobEventMessage& event) noexcept;
  void observe_workers(std::size_t busy, std::uint64_t failures) noexcept;
  void record_event_drops(std::uint64_t dropped) noexcept { total_.events_dropped = dropped; }

  [[nodiscard]] const JobMetricCounters& counters() const noexcept { return total_; }
  [[nodiscard]] const JobMetricCounters* counters(JobKind kind) const noexcept;
  [[nodiscard]] const LatencyHistogram& queue_latency() const noexcept { return queue_latency_; }
  [[nodiscard]] const LatencyHistogram& execution_latency() const noexcept
  {
    return execution_latency_;
  }
  [[nodiscard]] const LatencyHistogram& end_to_end_latency() const noexcept
  {
    return end_to_end_latency_;
  }
  [[nodiscard]] const LatencyHistogram& cancellation_latency() const noexcept
  {
    return cancellation_latency_;
  }

private:
  JobMetricCounters total_{};
  std::array<JobMetricCounters, 3uz> kinds_{};
  LatencyHistogram queue_latency_{};
  LatencyHistogram execution_latency_{};
  LatencyHistogram end_to_end_latency_{};
  LatencyHistogram cancellation_latency_{};
};

void write_prometheus(std::ostream& output, const RelayMetrics& metrics,
                      std::string_view prefix = "flowedge_relay");
void write_metrics_json(std::ostream& output, const RelayMetrics& metrics);
void write_otlp_json(std::ostream& output, const RelayMetrics& metrics,
                     std::string_view service_name = "flowedge-relayd");
void write_prometheus(std::ostream& output, const JobMetrics& metrics,
                      std::string_view prefix = "flowedge_relay_job");
void write_metrics_json(std::ostream& output, const JobMetrics& metrics);
void write_otlp_json(std::ostream& output, const JobMetrics& metrics,
                     std::string_view service_name = "flowedge-relay-jobs");
[[nodiscard]] std::expected<void, std::string> export_prometheus(
    std::string_view path, const RelayMetrics& metrics) noexcept;
[[nodiscard]] std::expected<void, std::string> export_metrics_json(
    std::string_view path, const RelayMetrics& metrics) noexcept;
[[nodiscard]] std::expected<void, std::string> export_otlp_json(
    std::string_view path, const RelayMetrics& metrics,
    std::string_view service_name = "flowedge-relayd") noexcept;
[[nodiscard]] std::expected<void, std::string> export_prometheus(
    std::string_view path, const JobMetrics& metrics) noexcept;
[[nodiscard]] std::expected<void, std::string> export_metrics_json(
    std::string_view path, const JobMetrics& metrics) noexcept;
[[nodiscard]] std::expected<void, std::string> export_otlp_json(
    std::string_view path, const JobMetrics& metrics,
    std::string_view service_name = "flowedge-relay-jobs") noexcept;

} // namespace fe::relay
