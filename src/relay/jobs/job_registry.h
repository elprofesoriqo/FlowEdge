#pragma once

#include "relay/jobs/cooperative_job.h"
#include "relay/protocol/job_messages.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string_view>

namespace fe::relay {

struct JobRoute
{
  JobKind kind{JobKind::kUnknown};
  ModelDigest model_digest{};
  std::uint64_t state_schema{};

  [[nodiscard]] friend bool operator==(const JobRoute&, const JobRoute&) = default;
};

[[nodiscard]] constexpr JobRoute job_route(const JobDescriptor& descriptor) noexcept
{
  return JobRoute{
      .kind = descriptor.kind,
      .model_digest = descriptor.model_digest,
      .state_schema = descriptor.state_schema,
  };
}

struct RoutedJobOps
{
  CooperativeJobOps cooperative{};
  bool (*prepare)(void* context, std::span<const std::byte> request) noexcept {};
  std::span<const std::byte> (*result)(const void* context) noexcept {};
};

// One registration binds a route to one caller-owned backend instance. A worker
// that may execute the same route concurrently provisions one instance and one
// registry entry per execution lane.
struct JobAdapterRegistration
{
  JobRoute route{};
  void* context{};
  RoutedJobOps operations{};
  std::size_t max_request_bytes{};
  std::size_t max_result_bytes{};
};

enum class JobRouteErrorCode : std::uint8_t
{
  kInvalidRegistration,
  kDuplicateRoute,
  kCapacityExceeded,
  kRegistryFrozen,
  kRegistryNotFrozen,
  kInvalidRequest,
  kAdapterNotFound,
  kPayloadTooLarge,
  kPrepareFailed,
  kInvalidBinding,
};

struct JobRouteError
{
  JobRouteErrorCode code{};
  std::string_view message{};
};

class RoutedJob
{
public:
  [[nodiscard]] const JobDescriptor& descriptor() const noexcept { return job_.descriptor(); }
  [[nodiscard]] JobProgress progress() const noexcept { return job_.progress(); }
  [[nodiscard]] std::expected<JobProgress, JobError> start() noexcept { return job_.start(); }
  [[nodiscard]] std::expected<JobProgress, JobError> advance(
      std::size_t work_budget, std::uint64_t cancel_before_generation = 0u) noexcept
  {
    return job_.advance(work_budget, cancel_before_generation);
  }
  [[nodiscard]] JobProgress cancel_before(std::uint64_t generation) noexcept
  {
    return job_.cancel_before(generation);
  }
  [[nodiscard]] std::size_t capsule_bytes() const noexcept { return job_.capsule_bytes(); }
  [[nodiscard]] std::expected<std::size_t, JobError> export_capsule(
      std::span<std::byte> destination) const noexcept
  {
    return job_.export_capsule(destination);
  }
  [[nodiscard]] std::expected<JobProgress, JobError> restore_capsule(
      std::span<const std::byte> source) noexcept
  {
    return job_.restore_capsule(source);
  }
  [[nodiscard]] ProtocolResult write_result(JobResultMessage& destination,
                                            std::uint64_t timestamp_ns) const noexcept;

private:
  friend class JobAdapterRegistry;
  RoutedJob(CooperativeJob job, std::uint64_t sequence, void* context,
            std::span<const std::byte> (*result)(const void*) noexcept,
            std::size_t max_result_bytes, JobServiceClass service_class) noexcept
      : job_{job}, sequence_{sequence}, context_{context}, result_{result},
        max_result_bytes_{max_result_bytes}, service_class_{service_class}
  {
  }

  CooperativeJob job_;
  std::uint64_t sequence_{};
  void* context_{};
  std::span<const std::byte> (*result_)(const void*) noexcept {};
  std::size_t max_result_bytes_{};
  JobServiceClass service_class_{JobServiceClass::kBestEffort};
};

// Fixed-capacity registry. Registration happens during initialization; freeze
// publishes immutable route metadata that can be looked up concurrently. Binding
// still requires exclusive ownership of the selected mutable backend lane. The
// caller owns the backing entries and every registered backend.
class JobAdapterRegistry
{
public:
  explicit JobAdapterRegistry(std::span<JobAdapterRegistration> storage) noexcept
      : storage_{storage}
  {
  }

  [[nodiscard]] std::expected<void, JobRouteError> add(
      JobAdapterRegistration registration) noexcept;
  void freeze() noexcept { frozen_ = true; }

  [[nodiscard]] bool frozen() const noexcept { return frozen_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] std::size_t capacity() const noexcept { return storage_.size(); }
  [[nodiscard]] const JobAdapterRegistration* find(const JobDescriptor& descriptor) const noexcept;
  [[nodiscard]] std::expected<RoutedJob, JobRouteError> bind(
      const JobRequestMessage& request) const noexcept;

private:
  std::span<JobAdapterRegistration> storage_{};
  std::size_t size_{};
  bool frozen_{};
};

} // namespace fe::relay
