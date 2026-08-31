#include "relay/jobs/job_registry.h"

#include <algorithm>

namespace fe::relay {
namespace {

[[nodiscard]] bool valid_operations(const RoutedJobOps& operations) noexcept
{
  const CooperativeJobOps& cooperative = operations.cooperative;
  return cooperative.begin != nullptr && cooperative.advance != nullptr &&
         cooperative.cancel != nullptr && cooperative.state_bytes != nullptr &&
         cooperative.save_state != nullptr && cooperative.load_state != nullptr &&
         operations.prepare != nullptr && operations.result != nullptr;
}

[[nodiscard]] bool valid_route(const JobRoute& route) noexcept
{
  bool has_digest{false};
  for (const std::uint8_t byte : route.model_digest)
    has_digest = has_digest || byte != 0u;
  return valid_job_kind(route.kind) && route.state_schema != 0u && has_digest;
}

} // namespace

std::expected<void, JobRouteError> JobAdapterRegistry::add(
    JobAdapterRegistration registration) noexcept
{
  if (frozen_)
    return std::unexpected(JobRouteError{JobRouteErrorCode::kRegistryFrozen,
                                         "Job adapter registry is already frozen"});
  if (!valid_route(registration.route) || registration.context == nullptr ||
      !valid_operations(registration.operations) ||
      registration.max_request_bytes > kMaxJobPayloadBytes ||
      registration.max_result_bytes > kMaxJobPayloadBytes) {
    return std::unexpected(JobRouteError{JobRouteErrorCode::kInvalidRegistration,
                                         "Job adapter registration is invalid"});
  }
  const auto entries = storage_.first(size_);
  if (std::ranges::find_if(entries, [&](const JobAdapterRegistration& entry) {
        return entry.route == registration.route;
      }) != entries.end()) {
    return std::unexpected(JobRouteError{JobRouteErrorCode::kDuplicateRoute,
                                         "Job adapter route is already registered"});
  }
  if (size_ == storage_.size())
    return std::unexpected(JobRouteError{JobRouteErrorCode::kCapacityExceeded,
                                         "Job adapter registry capacity is exhausted"});
  storage_[size_++] = registration;
  return {};
}

const JobAdapterRegistration* JobAdapterRegistry::find(
    const JobDescriptor& descriptor) const noexcept
{
  if (!frozen_ || !valid_job_descriptor(descriptor))
    return nullptr;
  const JobRoute route = job_route(descriptor);
  const auto entries = storage_.first(size_);
  const auto found = std::ranges::find_if(entries, [&](const JobAdapterRegistration& entry) {
    return entry.route == route;
  });
  return found == entries.end() ? nullptr : &*found;
}

std::expected<RoutedJob, JobRouteError> JobAdapterRegistry::bind(
    const JobRequestMessage& request) const noexcept
{
  if (!frozen_)
    return std::unexpected(JobRouteError{JobRouteErrorCode::kRegistryNotFrozen,
                                         "Job adapter registry must be frozen before routing"});
  if (validate(request) != ProtocolResult::kSuccess)
    return std::unexpected(
        JobRouteError{JobRouteErrorCode::kInvalidRequest, "Generic job request is invalid"});
  const JobAdapterRegistration* const registration = find(request.metadata.descriptor);
  if (registration == nullptr)
    return std::unexpected(JobRouteError{JobRouteErrorCode::kAdapterNotFound,
                                         "No adapter matches the job kind, model, and schema"});
  if (request.metadata.payload_bytes > registration->max_request_bytes)
    return std::unexpected(JobRouteError{JobRouteErrorCode::kPayloadTooLarge,
                                         "Generic job request exceeds the adapter payload limit"});
  if (!registration->operations.prepare(registration->context, request.payload_values()))
    return std::unexpected(JobRouteError{JobRouteErrorCode::kPrepareFailed,
                                         "Generic job adapter rejected the request payload"});

  const auto bound = CooperativeJob::bind(registration->context, request.metadata.descriptor,
                                          registration->operations.cooperative);
  if (!bound)
    return std::unexpected(JobRouteError{JobRouteErrorCode::kInvalidBinding,
                                         "Generic job adapter could not bind a cooperative job"});
  return RoutedJob{*bound, request.envelope.sequence, registration->context,
                   registration->operations.result, registration->max_result_bytes};
}

ProtocolResult RoutedJob::write_result(JobResultMessage& destination,
                                       std::uint64_t timestamp_ns) const noexcept
{
  const std::span<const std::byte> payload = result_(context_);
  if (payload.size() > max_result_bytes_ || payload.size() > kMaxJobPayloadBytes)
    return ProtocolResult::kDimensionExceeded;
  const JobProgress current = job_.progress();
  return make_job_result(destination, sequence_, job_.descriptor(), current, timestamp_ns,
                         result_code(current.state), payload);
}

} // namespace fe::relay
