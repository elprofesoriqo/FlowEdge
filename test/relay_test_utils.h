#include "relay/adapters/cooperative_adapters.h"
#include "relay/adapters/mamba_stream_adapter.h"
#include "relay/adapters/routed_adapters.h"
#include "relay/client/action_delivery.h"
#include "relay/client/job_client.h"
#include "relay/client/job_control_client.h"
#include "relay/jobs/state_capsule.h"
#include "relay/jobs/state_codec.h"
#include "relay/protocol/messages.h"
#include "relay/protocol/trace.h"
#include "relay/scheduler/edf_scheduler.h"
#include "relay/shared_memory/shared_memory_ring.h"
#include "relay/telemetry/metrics.h"
#include "relay/worker/head_worker.h"
#include "relay/worker/head_worker_pool.h"
#include "relay/worker/job_control_service.h"
#include "relay/worker/job_service.h"
#include "relay/worker/job_worker_pool.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <ranges>
#include <span>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace fe::relay;

static_assert(static_cast<std::uint16_t>(MessageKind::kCondition) == 1u);
static_assert(static_cast<std::uint16_t>(MessageKind::kAction) == 2u);
static_assert(static_cast<std::uint16_t>(MessageKind::kShutdown) == 3u);
static_assert(static_cast<std::uint16_t>(MessageKind::kJobControlRequest) == 7u);
static_assert(static_cast<std::uint16_t>(MessageKind::kJobControlResponse) == 8u);

[[nodiscard]] fe_model_metadata test_model(std::size_t condition_dim = 4uz,
                                           std::size_t action_dim = 2uz)
{
  fe_model_metadata model{};
  model.struct_size = sizeof(model);
  model.protocol_version = FE_PROTOCOL_VERSION;
  model.architecture = FE_ARCH_FLOW_HEAD;
  model.precision = FE_PRECISION_F32;
  model.condition_dim = condition_dim;
  model.action_dim = action_dim;
  for (std::size_t i{0uz}; i < FE_MODEL_DIGEST_BYTES; ++i)
    model.model_digest.bytes[i] = static_cast<std::uint8_t>(i + 1uz);
  return model;
}

[[nodiscard]] ConditionMessage request(std::uint64_t sequence, std::uint64_t generation,
                                       std::uint64_t deadline)
{
  ConditionMessage message{};
  constexpr std::array condition{1.0f, 2.0f, 3.0f, 4.0f};
  constexpr std::array noise{0.25f, -0.25f};
  EXPECT_EQ(make_condition_message(message, test_model(), sequence, 77u, 10u, deadline, generation,
                                   4uz, FE_SOLVER_HEUN, condition, noise),
            ProtocolResult::kSuccess);
  return message;
}

[[nodiscard]] ActionMessage complete_action(
    const fe_model_metadata& model, std::span<const float> values, std::uint64_t sequence = 1u,
    std::uint64_t session_id = 77u, std::uint64_t generation = 1u,
    std::uint64_t timestamp_ns = 100u, std::uint64_t deadline_ns = 0u)
{
  ActionMessage action{};
  action.envelope.kind = MessageKind::kAction;
  action.envelope.sequence = sequence;
  action.envelope.session_id = session_id;
  action.metadata.struct_size = sizeof(fe_action_metadata);
  action.metadata.protocol_version = FE_PROTOCOL_VERSION;
  action.metadata.solver = FE_SOLVER_EULER;
  action.metadata.status = FE_ACTION_COMPLETE;
  action.metadata.model_digest = model.model_digest;
  action.metadata.timestamp_ns = timestamp_ns;
  action.metadata.deadline_ns = deadline_ns;
  action.metadata.generation = generation;
  action.metadata.condition_dim = model.condition_dim;
  action.metadata.action_dim = model.action_dim;
  action.metadata.remaining_nfe = 0u;
  action.envelope.struct_size = static_cast<std::uint32_t>(wire_size(action));
  EXPECT_EQ(values.size(), model.action_dim);
  std::ranges::copy(values, action.action.begin());
  return action;
}

[[nodiscard]] std::string unique_name(std::string_view prefix)
{
  const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::string{prefix} + '-' + std::to_string(tick);
}

struct TestCooperativeBackend
{
  static constexpr std::size_t kStateBytes =
      sizeof(std::uint64_t) + (4uz * sizeof(std::uint32_t)) + sizeof(std::uint8_t);
  static constexpr std::uint64_t kTotalSteps = 6u;

  [[nodiscard]] bool begin() noexcept
  {
    step = 0u;
    values = {1.0f, 2.0f, 3.0f, 4.0f};
    cancelled = false;
    return true;
  }

  [[nodiscard]] BackendAdvance advance(std::size_t budget) noexcept
  {
    const std::size_t count =
        static_cast<std::size_t>(std::min<std::uint64_t>(budget, kTotalSteps - step));
    for (std::size_t unit{}; unit < count; ++unit) {
      ++step;
      for (std::size_t index{}; index < values.size(); ++index)
        values[index] = (values[index] * 1.125f) + static_cast<float>(step + index) * 0.03125f;
    }
    return BackendAdvance{
        .step = step == kTotalSteps ? BackendStep::kComplete : BackendStep::kInProgress,
        .completed_work_units = count,
    };
  }

  void cancel() noexcept { cancelled = true; }
  [[nodiscard]] std::size_t state_bytes() const noexcept { return kStateBytes; }

  [[nodiscard]] bool save_state(std::span<std::byte> destination) const noexcept
  {
    StateWriter writer{destination};
    if (!writer.write(step))
      return false;
    for (const float value : values) {
      if (!writer.write_float(value))
        return false;
    }
    return writer.write(static_cast<std::uint8_t>(cancelled)) && writer.remaining() == 0uz;
  }

  [[nodiscard]] bool load_state(std::span<const std::byte> source) noexcept
  {
    StateReader reader{source};
    const auto decoded_step = reader.read<std::uint64_t>();
    std::array<float, 4> decoded_values{};
    for (float& value : decoded_values) {
      const auto decoded = reader.read_float();
      if (!decoded)
        return false;
      value = *decoded;
    }
    const auto decoded_cancelled = reader.read<std::uint8_t>();
    if (!decoded_step || !decoded_cancelled || *decoded_step > kTotalSteps ||
        *decoded_cancelled > 1u || reader.remaining() != 0uz)
      return false;
    step = *decoded_step;
    values = decoded_values;
    cancelled = *decoded_cancelled != 0u;
    return true;
  }

  std::uint64_t step{};
  std::array<float, 4> values{};
  bool cancelled{};
};

struct TestRoutedBackend
{
  [[nodiscard]] bool prepare(std::span<const std::byte> request) noexcept
  {
    StateReader reader{request};
    const auto decoded = reader.read<std::uint64_t>();
    if (!decoded || reader.remaining() != 0uz)
      return false;
    seed = *decoded;
    return true;
  }

  [[nodiscard]] bool begin() noexcept
  {
    completed = 0u;
    value = seed;
    update_result();
    return true;
  }

  [[nodiscard]] BackendAdvance advance(std::size_t budget) noexcept
  {
    const std::size_t count =
        static_cast<std::size_t>(std::min<std::uint64_t>(budget, kTotalSteps - completed));
    for (std::size_t index{}; index < count; ++index) {
      value = (value * 3u) + completed + 1u;
      ++completed;
    }
    update_result();
    return BackendAdvance{
        .step = completed == kTotalSteps ? BackendStep::kComplete : BackendStep::kInProgress,
        .completed_work_units = count,
    };
  }

  void cancel() noexcept {}
  [[nodiscard]] std::size_t state_bytes() const noexcept { return 16uz; }

  [[nodiscard]] bool save_state(std::span<std::byte> destination) const noexcept
  {
    StateWriter writer{destination};
    return writer.write(completed) && writer.write(value) && writer.remaining() == 0uz;
  }

  [[nodiscard]] bool load_state(std::span<const std::byte> source) noexcept
  {
    StateReader reader{source};
    const auto next_completed = reader.read<std::uint64_t>();
    const auto next_value = reader.read<std::uint64_t>();
    if (!next_completed || !next_value || *next_completed > kTotalSteps ||
        reader.remaining() != 0uz)
      return false;
    completed = *next_completed;
    value = *next_value;
    update_result();
    return true;
  }

  [[nodiscard]] std::span<const std::byte> result() const noexcept { return encoded_result; }

  void update_result() noexcept
  {
    StateWriter writer{encoded_result};
    static_cast<void>(writer.write(value));
  }

  static constexpr std::uint64_t kTotalSteps = 4u;
  std::uint64_t seed{};
  std::uint64_t completed{};
  std::uint64_t value{};
  std::array<std::byte, sizeof(std::uint64_t)> encoded_result{};
};

static_assert(RoutedBackend<TestRoutedBackend>);

struct FailingRoutedBackend
{
  [[nodiscard]] bool prepare(std::span<const std::byte>) noexcept { return true; }
  [[nodiscard]] bool begin() noexcept { return false; }
  [[nodiscard]] BackendAdvance advance(std::size_t) noexcept
  {
    return {.step = BackendStep::kFailed, .completed_work_units = 0uz};
  }
  void cancel() noexcept {}
  [[nodiscard]] std::size_t state_bytes() const noexcept { return 1uz; }
  [[nodiscard]] bool save_state(std::span<std::byte>) const noexcept { return false; }
  [[nodiscard]] bool load_state(std::span<const std::byte>) noexcept { return false; }
  [[nodiscard]] std::span<const std::byte> result() const noexcept { return {}; }
};

static_assert(RoutedBackend<FailingRoutedBackend>);

struct DrainBackend
{
  [[nodiscard]] bool prepare(std::span<const std::byte> request) noexcept
  {
    StateReader reader{request};
    const auto decoded = reader.read<std::uint64_t>();
    if (!decoded || reader.remaining() != 0uz)
      return false;
    seed = *decoded;
    return true;
  }

  [[nodiscard]] bool begin() noexcept
  {
    completed = 0u;
    value = seed;
    update_result();
    return true;
  }

  [[nodiscard]] BackendAdvance advance(std::size_t budget) noexcept
  {
    const std::size_t count =
        static_cast<std::size_t>(std::min<std::uint64_t>(budget, kTotalSteps - completed));
    for (std::size_t index{}; index < count; ++index) {
      value = (value * 5u) + completed + 1u;
      ++completed;
      if (gate_first_boundary && completed == 1u) {
        boundary_reached->store(true, std::memory_order_release);
        boundary_reached->notify_one();
        bool closed{false};
        while (!release_boundary->load(std::memory_order_acquire))
          release_boundary->wait(closed, std::memory_order_relaxed);
      }
    }
    update_result();
    return {.step = completed == kTotalSteps ? BackendStep::kComplete : BackendStep::kInProgress,
            .completed_work_units = count};
  }

  void cancel() noexcept {}
  [[nodiscard]] std::size_t state_bytes() const noexcept { return 16uz; }
  [[nodiscard]] bool save_state(std::span<std::byte> destination) const noexcept
  {
    StateWriter writer{destination};
    return writer.write(completed) && writer.write(value) && writer.remaining() == 0uz;
  }
  [[nodiscard]] bool load_state(std::span<const std::byte> source) noexcept
  {
    StateReader reader{source};
    const auto next_completed = reader.read<std::uint64_t>();
    const auto next_value = reader.read<std::uint64_t>();
    if (!next_completed || !next_value || *next_completed > kTotalSteps ||
        reader.remaining() != 0uz)
      return false;
    completed = *next_completed;
    value = *next_value;
    update_result();
    return true;
  }
  [[nodiscard]] std::span<const std::byte> result() const noexcept { return encoded_result; }

  void update_result() noexcept
  {
    StateWriter writer{encoded_result};
    static_cast<void>(writer.write(value));
  }

  static constexpr std::uint64_t kTotalSteps = 6u;
  std::atomic_bool* boundary_reached{};
  std::atomic_bool* release_boundary{};
  bool gate_first_boundary{};
  std::uint64_t seed{};
  std::uint64_t completed{};
  std::uint64_t value{};
  std::array<std::byte, sizeof(std::uint64_t)> encoded_result{};
};

static_assert(RoutedBackend<DrainBackend>);

[[nodiscard]] JobDescriptor test_job_descriptor(std::uint64_t generation = 7u)
{
  JobDescriptor descriptor{};
  const fe_model_metadata model = test_model();
  for (std::size_t index{}; index < descriptor.model_digest.size(); ++index)
    descriptor.model_digest[index] = model.model_digest.bytes[index];
  descriptor.state_schema = 0x746f792d7631u; // "toy-v1"
  descriptor.session_id = 44u;
  descriptor.generation = generation;
  descriptor.deadline_ns = 1'000'000u;
  descriptor.total_work_units = TestCooperativeBackend::kTotalSteps;
  return descriptor;
}
