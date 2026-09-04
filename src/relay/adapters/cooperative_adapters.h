#pragma once

#include "relay/jobs/cooperative_job.h"

#include <concepts>
#include <cstddef>
#include <expected>
#include <span>

namespace fe::relay {

template<typename Backend>
concept CooperativeBackend =
    requires(Backend& backend, const Backend& const_backend, std::size_t budget,
             std::span<std::byte> destination, std::span<const std::byte> source) {
      { backend.begin() } noexcept -> std::same_as<bool>;
      { backend.advance(budget) } noexcept -> std::same_as<BackendAdvance>;
      { backend.cancel() } noexcept -> std::same_as<void>;
      { const_backend.state_bytes() } noexcept -> std::same_as<std::size_t>;
      { const_backend.save_state(destination) } noexcept -> std::same_as<bool>;
      { backend.load_state(source) } noexcept -> std::same_as<bool>;
    };

namespace detail {

template<CooperativeBackend Backend> struct CooperativeAdapterBridge
{
  [[nodiscard]] static bool begin(void* context) noexcept
  {
    return static_cast<Backend*>(context)->begin();
  }

  [[nodiscard]] static BackendAdvance advance(void* context, std::size_t budget) noexcept
  {
    return static_cast<Backend*>(context)->advance(budget);
  }

  static void cancel(void* context) noexcept { static_cast<Backend*>(context)->cancel(); }

  [[nodiscard]] static std::size_t state_bytes(const void* context) noexcept
  {
    return static_cast<const Backend*>(context)->state_bytes();
  }

  [[nodiscard]] static bool save_state(const void* context,
                                       std::span<std::byte> destination) noexcept
  {
    return static_cast<const Backend*>(context)->save_state(destination);
  }

  [[nodiscard]] static bool load_state(void* context, std::span<const std::byte> source) noexcept
  {
    return static_cast<Backend*>(context)->load_state(source);
  }

  [[nodiscard]] static constexpr CooperativeJobOps operations() noexcept
  {
    return CooperativeJobOps{
        .begin = begin,
        .advance = advance,
        .cancel = cancel,
        .state_bytes = state_bytes,
        .save_state = save_state,
        .load_state = load_state,
    };
  }
};

template<CooperativeBackend Backend>
[[nodiscard]] std::expected<CooperativeJob, JobError> make_adapted_job(Backend& backend,
                                                                       JobDescriptor descriptor,
                                                                       JobKind kind) noexcept
{
  descriptor.kind = kind;
  return CooperativeJob::bind(&backend, descriptor,
                              CooperativeAdapterBridge<Backend>::operations());
}

} // namespace detail

// One work unit is one complete solver, denoising, planning, or refinement step.
template<CooperativeBackend Backend>
[[nodiscard]] std::expected<CooperativeJob, JobError> make_iterative_job(
    Backend& backend, JobDescriptor descriptor) noexcept
{
  return detail::make_adapted_job(backend, descriptor, JobKind::kIterative);
}

// One work unit is one complete token, recurrence, or stream-chunk update.
template<CooperativeBackend Backend>
[[nodiscard]] std::expected<CooperativeJob, JobError> make_streaming_job(
    Backend& backend, JobDescriptor descriptor) noexcept
{
  return detail::make_adapted_job(backend, descriptor, JobKind::kStreaming);
}

// One work unit is one bounded draft or verification quantum. Branch state and
// accepted-prefix rollback remain owned by the adapted backend.
template<CooperativeBackend Backend>
[[nodiscard]] std::expected<CooperativeJob, JobError> make_speculative_job(
    Backend& backend, JobDescriptor descriptor) noexcept
{
  return detail::make_adapted_job(backend, descriptor, JobKind::kSpeculative);
}

} // namespace fe::relay
