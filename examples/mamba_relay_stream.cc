#include "relay/adapters/cooperative_adapters.h"
#include "relay/adapters/mamba_stream_adapter.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <utility>
#include <vector>

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cerr << "usage: mamba_relay_stream <model.safetensors>\n";
    return 2;
  }

  constexpr std::array<std::int32_t, 4> tokens{1, 2, 3, 4};
  std::unique_ptr<fe_weights, decltype(&fe_weights_free)> weights{fe_weights_load(argv[1]),
                                                                  fe_weights_free};
  if (!weights) {
    std::cerr << fe_engine_last_error() << '\n';
    return 1;
  }
  auto opened_source = fe::relay::MambaStreamAdapter::open(weights.get(), tokens.size(), 0u);
  auto opened_destination = fe::relay::MambaStreamAdapter::open(weights.get(), tokens.size(), 0u);
  if (!opened_source || !opened_destination) {
    std::cerr << (!opened_source ? opened_source.error() : opened_destination.error()) << '\n';
    return 1;
  }
  fe::relay::MambaStreamAdapter source = std::move(*opened_source);
  fe::relay::MambaStreamAdapter destination = std::move(*opened_destination);
  weights.reset(); // Engines retain the shared immutable checkpoint.

  std::array<std::byte, tokens.size() * sizeof(std::int32_t)> request_storage{};
  const auto encoded = fe::relay::encode_mamba_stream_request(tokens, request_storage);
  if (!encoded) {
    std::cerr << encoded.error() << '\n';
    return 1;
  }
  const auto request = std::span{request_storage}.first(*encoded);
  const auto descriptor = source.make_descriptor(1u, 1u, tokens.size());
  if (!source.prepare(request))
    return 1;
  auto bound_source = fe::relay::make_streaming_job(source, descriptor);
  if (!bound_source)
    return 1;
  fe::relay::CooperativeJob source_job = *bound_source;
  if (!source_job.start() || !source_job.advance(2uz))
    return 1;

  std::vector<std::byte> capsule(source_job.capsule_bytes());
  const auto exported = source_job.export_capsule(capsule);
  if (!exported)
    return 1;

  auto bound_destination = fe::relay::make_streaming_job(destination, descriptor);
  if (!bound_destination)
    return 1;
  fe::relay::CooperativeJob destination_job = *bound_destination;
  const auto restored = destination_job.restore_capsule(capsule);
  if (!restored)
    return 1;
  const auto completed = destination_job.advance(tokens.size() - 2uz);
  if (!completed || completed->state != fe::relay::JobState::kComplete)
    return 1;

  const std::span<const std::byte> output = destination.result();
  float first_hidden{};
  std::memcpy(&first_hidden, output.data(), sizeof(first_hidden));
  std::cout << "tokens=" << tokens.size() << " migrated_after=2 capsule_bytes=" << *exported
            << " result_bytes=" << output.size() << " hidden0=" << first_hidden << '\n';
  return 0;
}
