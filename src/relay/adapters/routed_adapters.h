#pragma once

#include "relay/adapters/cooperative_adapters.h"
#include "relay/jobs/job_registry.h"

#include <concepts>
#include <cstddef>
#include <span>

namespace fe::relay {

template<typename Backend>
concept RoutedBackend =
    CooperativeBackend<Backend> &&
    requires(Backend& backend, const Backend& const_backend, std::span<const std::byte> request) {
      { backend.prepare(request) } noexcept -> std::same_as<bool>;
      { const_backend.result() } noexcept -> std::same_as<std::span<const std::byte>>;
    };

namespace detail {

template<RoutedBackend Backend> struct RoutedAdapterBridge
{
  [[nodiscard]] static bool prepare(void* context, std::span<const std::byte> request) noexcept
  {
    return static_cast<Backend*>(context)->prepare(request);
  }

  [[nodiscard]] static std::span<const std::byte> result(const void* context) noexcept
  {
    return static_cast<const Backend*>(context)->result();
  }
};

} // namespace detail

template<RoutedBackend Backend>
[[nodiscard]] JobAdapterRegistration make_routed_adapter(Backend& backend, JobRoute route,
                                                         std::size_t max_request_bytes,
                                                         std::size_t max_result_bytes) noexcept
{
  return JobAdapterRegistration{
      .route = route,
      .context = &backend,
      .operations =
          RoutedJobOps{
              .cooperative = detail::CooperativeAdapterBridge<Backend>::operations(),
              .prepare = detail::RoutedAdapterBridge<Backend>::prepare,
              .result = detail::RoutedAdapterBridge<Backend>::result,
          },
      .max_request_bytes = max_request_bytes,
      .max_result_bytes = max_result_bytes,
  };
}

} // namespace fe::relay
