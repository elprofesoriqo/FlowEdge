#include "relay/adapters/cooperative_adapters.h"
#include "relay/adapters/mamba_stream_adapter.h"
#include "relay/adapters/routed_adapters.h"
#include "relay/client/job_client.h"
#include "relay/jobs/state_capsule.h"
#include "relay/jobs/state_codec.h"
#include "relay/protocol/messages.h"
#include "relay/protocol/trace.h"
#include "relay/scheduler/edf_scheduler.h"
#include "relay/shared_memory/shared_memory_ring.h"
#include "relay/telemetry/metrics.h"
#include "relay/worker/head_worker.h"
#include "relay/worker/head_worker_pool.h"
#include "relay/worker/job_service.h"
#include "relay/worker/job_worker_pool.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
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

TEST(StateCodec, PreservesCanonicalLittleEndianBits)
{
  std::array<std::byte, 12> encoded{};
  StateWriter writer{encoded};
  ASSERT_TRUE(writer.write(std::uint32_t{0x78563412u}));
  ASSERT_TRUE(writer.write(std::int32_t{-2}));
  ASSERT_TRUE(writer.write_float(1.0F));
  EXPECT_EQ(std::to_integer<std::uint8_t>(encoded[0]), 0x12u);
  EXPECT_EQ(std::to_integer<std::uint8_t>(encoded[3]), 0x78u);
  EXPECT_EQ(std::to_integer<std::uint8_t>(encoded[4]), 0xfeu);
  EXPECT_EQ(std::to_integer<std::uint8_t>(encoded[7]), 0xffu);
  EXPECT_EQ(std::to_integer<std::uint8_t>(encoded[10]), 0x80u);
  EXPECT_EQ(std::to_integer<std::uint8_t>(encoded[11]), 0x3fu);

  StateReader reader{encoded};
  EXPECT_EQ(reader.read<std::uint32_t>(), 0x78563412u);
  EXPECT_EQ(reader.read<std::int32_t>(), -2);
  EXPECT_EQ(reader.read_float(), 1.0F);
  EXPECT_FALSE(reader.read<std::uint8_t>());
}

TEST(RelayProtocol, BuildsAndValidatesVersionedCondition)
{
  const ConditionMessage message = request(9u, 3u, 100u);
  EXPECT_EQ(validate(message), ProtocolResult::kSuccess);
  EXPECT_EQ(message.envelope.sequence, 9u);
  EXPECT_EQ(message.envelope.session_id, 77u);
  EXPECT_EQ(message.metadata.generation, 3u);
  EXPECT_EQ(message.metadata.remaining_nfe, 8u);
  EXPECT_FLOAT_EQ(message.condition_values()[3], 4.0f);
  EXPECT_FLOAT_EQ(message.noise_values()[1], -0.25f);
  EXPECT_TRUE(compatible(message, test_model()));

  ConditionMessage invalid = message;
  invalid.envelope.version += 1u;
  EXPECT_EQ(validate(invalid), ProtocolResult::kInvalidEnvelope);
  invalid = message;
  invalid.metadata.action_dim = kMaxActionDim + 1uz;
  EXPECT_EQ(validate(invalid), ProtocolResult::kDimensionExceeded);
  invalid = message;
  invalid.metadata.remaining_nfe += 1u;
  EXPECT_EQ(validate(invalid), ProtocolResult::kInvalidMetadata);
  auto mismatched = test_model();
  mismatched.model_digest.bytes[0] ^= 0xffu;
  EXPECT_FALSE(compatible(message, mismatched));

  ActionMessage rejected{};
  make_rejected_action(rejected, message, 101u, RelayActionCode::kRejectedDeadline);
  EXPECT_EQ(validate(rejected), ProtocolResult::kSuccess);
  EXPECT_TRUE(compatible(rejected, test_model()));
  EXPECT_EQ(action_code(rejected), RelayActionCode::kRejectedDeadline);
  EXPECT_EQ(rejected.metadata.status, FE_ACTION_FAILED);
  EXPECT_EQ(rejected.metadata.remaining_nfe, message.metadata.remaining_nfe);
}

TEST(GenericJobRouting, ValidatesMessagesAndRoutesThroughFrozenRegistry)
{
  JobDescriptor descriptor = test_job_descriptor();
  descriptor.kind = JobKind::kIterative;
  descriptor.total_work_units = TestRoutedBackend::kTotalSteps;
  descriptor.deadline_ns = 1'000u;

  std::array<std::byte, sizeof(std::uint64_t)> input{};
  StateWriter input_writer{input};
  ASSERT_TRUE(input_writer.write(std::uint64_t{5u}));
  auto request_message = std::make_unique<JobRequestMessage>();
  ASSERT_EQ(make_job_request(*request_message, 91u, descriptor, 100u, input),
            ProtocolResult::kSuccess);
  EXPECT_EQ(validate(*request_message), ProtocolResult::kSuccess);
  EXPECT_EQ(request_message->envelope.kind, MessageKind::kJobRequest);
  EXPECT_EQ(wire_size(*request_message), offsetof(JobRequestMessage, payload) + input.size());

  JobRequestMessage invalid = *request_message;
  invalid.envelope.session_id += 1u;
  EXPECT_EQ(validate(invalid), ProtocolResult::kInvalidMetadata);

  TestRoutedBackend backend{};
  std::array<JobAdapterRegistration, 1> storage{};
  JobAdapterRegistry registry{storage};
  const JobAdapterRegistration registration =
      make_routed_adapter(backend, job_route(descriptor), input.size(), sizeof(std::uint64_t));
  ASSERT_TRUE(registry.add(registration));
  const auto duplicate = registry.add(registration);
  ASSERT_FALSE(duplicate);
  EXPECT_EQ(duplicate.error().code, JobRouteErrorCode::kDuplicateRoute);
  const auto before_freeze = registry.bind(*request_message);
  ASSERT_FALSE(before_freeze);
  EXPECT_EQ(before_freeze.error().code, JobRouteErrorCode::kRegistryNotFrozen);

  registry.freeze();
  ASSERT_NE(registry.find(descriptor), nullptr);
  const auto after_freeze = registry.add(registration);
  ASSERT_FALSE(after_freeze);
  EXPECT_EQ(after_freeze.error().code, JobRouteErrorCode::kRegistryFrozen);

  auto routed_result = registry.bind(*request_message);
  ASSERT_TRUE(routed_result) << routed_result.error().message;
  RoutedJob routed = *routed_result;
  ASSERT_TRUE(routed.start());
  ASSERT_TRUE(routed.advance(2uz));
  auto result_message = std::make_unique<JobResultMessage>();
  EXPECT_EQ(routed.write_result(*result_message, 200u), ProtocolResult::kSuccess);
  EXPECT_EQ(validate(*result_message), ProtocolResult::kSuccess);
  EXPECT_EQ(job_result_code(*result_message), JobResultCode::kProgress);
  EXPECT_EQ(result_message->metadata.progress.completed_work_units, 2u);

  ASSERT_TRUE(routed.advance(2uz));
  EXPECT_EQ(routed.write_result(*result_message, 300u), ProtocolResult::kSuccess);
  EXPECT_EQ(validate(*result_message), ProtocolResult::kSuccess);
  EXPECT_EQ(job_result_code(*result_message), JobResultCode::kComplete);
  StateReader result_reader{result_message->payload_values()};
  const auto value = result_reader.read<std::uint64_t>();
  ASSERT_TRUE(value);
  EXPECT_EQ(*value, backend.value);

  JobDescriptor missing = descriptor;
  ++missing.state_schema;
  ASSERT_EQ(make_job_request(*request_message, 92u, missing, 100u, input),
            ProtocolResult::kSuccess);
  const auto not_found = registry.bind(*request_message);
  ASSERT_FALSE(not_found);
  EXPECT_EQ(not_found.error().code, JobRouteErrorCode::kAdapterNotFound);

  TestRoutedBackend limited_backend{};
  std::array<JobAdapterRegistration, 1> limited_storage{};
  JobAdapterRegistry limited{limited_storage};
  ASSERT_TRUE(limited.add(
      make_routed_adapter(limited_backend, job_route(descriptor), 4uz, sizeof(std::uint64_t))));
  JobRoute another_route = job_route(descriptor);
  ++another_route.state_schema;
  const auto capacity = limited.add(
      make_routed_adapter(limited_backend, another_route, input.size(), sizeof(std::uint64_t)));
  ASSERT_FALSE(capacity);
  EXPECT_EQ(capacity.error().code, JobRouteErrorCode::kCapacityExceeded);
  limited.freeze();
  ASSERT_EQ(make_job_request(*request_message, 93u, descriptor, 100u, input),
            ProtocolResult::kSuccess);
  const auto rejected_payload = limited.bind(*request_message);
  ASSERT_FALSE(rejected_payload);
  EXPECT_EQ(rejected_payload.error().code, JobRouteErrorCode::kPayloadTooLarge);

  const std::string ring_name = unique_name("flowedge-generic-job-ring-test");
  constexpr std::size_t slot_bytes = std::max(sizeof(JobRequestMessage), sizeof(JobResultMessage));
  auto created = SharedMemoryRing::create(ring_name, RingConfig{2u, slot_bytes});
  ASSERT_TRUE(created) << created.error();
  auto opened = SharedMemoryRing::open(ring_name);
  ASSERT_TRUE(opened) << opened.error();
  SharedMemoryRing producer = std::move(*created);
  SharedMemoryRing consumer = std::move(*opened);
  EXPECT_EQ(producer.try_push(wire_bytes(*request_message)), RingResult::kSuccess);
  auto received = std::make_unique<JobRequestMessage>();
  std::size_t received_bytes{};
  EXPECT_EQ(consumer.try_pop(std::as_writable_bytes(std::span{received.get(), 1uz}),
                             received_bytes),
            RingResult::kSuccess);
  EXPECT_EQ(received_bytes, wire_size(*request_message));
  EXPECT_EQ(validate(*received), ProtocolResult::kSuccess);
  EXPECT_EQ(producer.try_push(wire_bytes(*result_message)), RingResult::kSuccess);
  auto received_result = std::make_unique<JobResultMessage>();
  EXPECT_EQ(consumer.try_pop(std::as_writable_bytes(std::span{received_result.get(), 1uz}),
                             received_bytes),
            RingResult::kSuccess);
  EXPECT_EQ(received_bytes, wire_size(*result_message));
  EXPECT_EQ(validate(*received_result), ProtocolResult::kSuccess);
}

TEST(JobWorkerPool, RoutesJobsAcrossPreallocatedAdapterLanes)
{
  JobDescriptor first_descriptor = test_job_descriptor(7u);
  first_descriptor.kind = JobKind::kIterative;
  first_descriptor.deadline_ns = 0u;
  first_descriptor.total_work_units = TestRoutedBackend::kTotalSteps;
  JobDescriptor second_descriptor = first_descriptor;
  second_descriptor.session_id += 1u;

  TestRoutedBackend first_backend{};
  TestRoutedBackend second_backend{};
  std::array<JobAdapterRegistration, 1> first_entries{};
  std::array<JobAdapterRegistration, 1> second_entries{};
  std::array<JobAdapterRegistry, 2> registries{JobAdapterRegistry{first_entries},
                                               JobAdapterRegistry{second_entries}};
  ASSERT_TRUE(registries[0].add(make_routed_adapter(first_backend, job_route(first_descriptor),
                                                    sizeof(std::uint64_t), sizeof(std::uint64_t))));
  ASSERT_TRUE(registries[1].add(make_routed_adapter(second_backend, job_route(first_descriptor),
                                                    sizeof(std::uint64_t), sizeof(std::uint64_t))));
  for (JobAdapterRegistry& registry : registries)
    registry.freeze();

  auto events = JobEventBuffer::create(16uz);
  ASSERT_TRUE(events) << events.error();
  auto created =
      JobWorkerPool::create(registries, 4uz, {}, 4uz, 1uz, WorkerPlacement::kSpread, &*events);
  ASSERT_TRUE(created) << created.error();
  JobWorkerPool pool = std::move(*created);

  std::array<std::byte, sizeof(std::uint64_t)> first_input{};
  std::array<std::byte, sizeof(std::uint64_t)> second_input{};
  StateWriter first_writer{first_input};
  StateWriter second_writer{second_input};
  ASSERT_TRUE(first_writer.write(std::uint64_t{5u}));
  ASSERT_TRUE(second_writer.write(std::uint64_t{11u}));
  auto first_request = std::make_unique<JobRequestMessage>();
  auto second_request = std::make_unique<JobRequestMessage>();
  ASSERT_EQ(make_job_request(*first_request, 101u, first_descriptor, 0u, first_input),
            ProtocolResult::kSuccess);
  ASSERT_EQ(make_job_request(*second_request, 102u, second_descriptor, 0u, second_input),
            ProtocolResult::kSuccess);
  EXPECT_EQ(pool.submit(*first_request), JobSubmitResult::kAccepted);
  EXPECT_EQ(pool.submit(*second_request), JobSubmitResult::kAccepted);

  std::array<bool, 2> received{};
  std::size_t completed{};
  const auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (completed < received.size() && std::chrono::steady_clock::now() < timeout) {
    const JobResultMessage* const result = pool.ready_result();
    if (result == nullptr) {
      std::this_thread::yield();
      continue;
    }
    ASSERT_EQ(validate(*result), ProtocolResult::kSuccess);
    EXPECT_EQ(job_result_code(*result), JobResultCode::kComplete);
    EXPECT_EQ(result->metadata.progress.completed_work_units, TestRoutedBackend::kTotalSteps);
    ASSERT_GE(result->envelope.sequence, 101u);
    ASSERT_LE(result->envelope.sequence, 102u);
    const std::size_t index = static_cast<std::size_t>(result->envelope.sequence - 101u);
    EXPECT_FALSE(received[index]);
    received[index] = true;
    ++completed;
    pool.release_ready_result();
  }
  EXPECT_EQ(completed, received.size());
  EXPECT_EQ(pool.failure_count(), 0u);
  EXPECT_EQ(pool.busy_count(), 0uz);
  EXPECT_EQ(pool.queued_count(), 0uz);
  JobMetrics metrics{};
  while (const JobEventMessage* event = events->front()) {
    EXPECT_EQ(validate(*event), ProtocolResult::kSuccess);
    metrics.record(*event);
    events->pop();
  }
  EXPECT_EQ(metrics.counters().events, 8u);
  EXPECT_EQ(metrics.counters().admitted, 2u);
  EXPECT_EQ(metrics.counters().dispatched, 2u);
  EXPECT_EQ(metrics.counters().started, 2u);
  EXPECT_EQ(metrics.counters().completed, 2u);
  EXPECT_EQ(metrics.counters().completed_work_units, 8u);
  ASSERT_NE(metrics.counters(JobKind::kIterative), nullptr);
  EXPECT_EQ(metrics.counters(JobKind::kIterative)->completed, 2u);
  EXPECT_EQ(events->dropped(), 0u);
  EXPECT_TRUE(pool.release_session(first_descriptor.session_id));
  EXPECT_TRUE(pool.release_session(second_descriptor.session_id));
}

TEST(JobWorkerPool, AppliesPerKindAdmissionAndSessionFreshness)
{
  JobDescriptor descriptor = test_job_descriptor(8u);
  descriptor.kind = JobKind::kIterative;
  descriptor.deadline_ns = 0u;
  descriptor.total_work_units = TestRoutedBackend::kTotalSteps;
  TestRoutedBackend backend{};
  std::array<JobAdapterRegistration, 1> entries{};
  std::array<JobAdapterRegistry, 1> registries{JobAdapterRegistry{entries}};
  ASSERT_TRUE(registries[0].add(make_routed_adapter(backend, job_route(descriptor),
                                                    sizeof(std::uint64_t), sizeof(std::uint64_t))));
  registries[0].freeze();
  auto created = JobWorkerPool::create(registries, 2uz,
                                       JobCostPolicy{.iterative_ns = 100u,
                                                     .streaming_ns = 10u,
                                                     .speculative_ns = 20u},
                                       2uz);
  ASSERT_TRUE(created) << created.error();
  JobWorkerPool pool = std::move(*created);

  std::array<std::byte, sizeof(std::uint64_t)> input{};
  StateWriter writer{input};
  ASSERT_TRUE(writer.write(std::uint64_t{3u}));
  auto request_message = std::make_unique<JobRequestMessage>();
  auto rejection = std::make_unique<JobResultMessage>();
  constexpr std::uint64_t now_ns = 1'000u;
  descriptor.deadline_ns = 0u;
  ASSERT_EQ(make_job_request(*request_message, 200u, descriptor, now_ns, input),
            ProtocolResult::kSuccess);
  request_message->envelope.magic = 0u;
  EXPECT_EQ(pool.submit(*request_message, now_ns, rejection.get()), JobSubmitResult::kInvalid);
  EXPECT_EQ(validate(*rejection), ProtocolResult::kSuccess);
  EXPECT_EQ(job_result_code(*rejection), JobResultCode::kInvalidRequest);

  descriptor.deadline_ns = now_ns + 399u;
  ASSERT_EQ(make_job_request(*request_message, 201u, descriptor, now_ns, input),
            ProtocolResult::kSuccess);
  EXPECT_EQ(pool.submit(*request_message, now_ns, rejection.get()),
            JobSubmitResult::kDeadlineUnreachable);
  EXPECT_EQ(validate(*rejection), ProtocolResult::kSuccess);
  EXPECT_EQ(job_result_code(*rejection), JobResultCode::kRejectedDeadline);

  descriptor.deadline_ns = 0u;
  ASSERT_EQ(make_job_request(*request_message, 202u, descriptor, 0u, input),
            ProtocolResult::kSuccess);
  EXPECT_EQ(pool.submit(*request_message), JobSubmitResult::kAccepted);
  EXPECT_FALSE(pool.release_session(descriptor.session_id));
  const auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  const JobResultMessage* completed{};
  while (completed == nullptr && std::chrono::steady_clock::now() < timeout) {
    completed = pool.ready_result();
    if (completed == nullptr)
      std::this_thread::yield();
  }
  ASSERT_NE(completed, nullptr);
  EXPECT_EQ(job_result_code(*completed), JobResultCode::kComplete);
  pool.release_ready_result();

  --descriptor.generation;
  ASSERT_EQ(make_job_request(*request_message, 203u, descriptor, 0u, input),
            ProtocolResult::kSuccess);
  EXPECT_EQ(pool.submit(*request_message, 0u, rejection.get()), JobSubmitResult::kStale);
  EXPECT_EQ(job_result_code(*rejection), JobResultCode::kRejectedStale);

  descriptor.generation += 2u;
  ++descriptor.state_schema;
  ASSERT_EQ(make_job_request(*request_message, 204u, descriptor, 0u, input),
            ProtocolResult::kSuccess);
  EXPECT_EQ(pool.submit(*request_message, 0u, rejection.get()), JobSubmitResult::kAdapterNotFound);
  EXPECT_EQ(job_result_code(*rejection), JobResultCode::kAdapterNotFound);
}

TEST(JobTransport, PreservesTypedResultsAcrossOutputBackpressure)
{
  JobDescriptor descriptor = test_job_descriptor(11u);
  descriptor.kind = JobKind::kIterative;
  descriptor.deadline_ns = 0u;
  descriptor.total_work_units = TestRoutedBackend::kTotalSteps;

  TestRoutedBackend backend{};
  std::array<JobAdapterRegistration, 1> entries{};
  std::array<JobAdapterRegistry, 1> registries{JobAdapterRegistry{entries}};
  ASSERT_TRUE(registries[0].add(make_routed_adapter(backend, job_route(descriptor),
                                                    sizeof(std::uint64_t), sizeof(std::uint64_t))));
  registries[0].freeze();
  auto created_pool = JobWorkerPool::create(registries, 4uz, {}, 4uz, 1uz);
  ASSERT_TRUE(created_pool) << created_pool.error();
  JobWorkerPool pool = std::move(*created_pool);

  const std::string request_name = unique_name("flowedge-job-service-requests");
  const std::string result_name = unique_name("flowedge-job-service-results");
  auto created_service = JobService::create(request_name, result_name, pool, 2u);
  ASSERT_TRUE(created_service) << created_service.error();
  JobService service = std::move(*created_service);
  auto connected = JobClient::connect(request_name, result_name);
  ASSERT_TRUE(connected) << connected.error();
  JobClient client = std::move(*connected);

  std::array<std::byte, sizeof(std::uint64_t)> input{};
  StateWriter writer{input};
  ASSERT_TRUE(writer.write(std::uint64_t{5u}));
  auto request_message = std::make_unique<JobRequestMessage>();
  ASSERT_EQ(make_job_request(*request_message, 500u, descriptor, 0u, input),
            ProtocolResult::kSuccess);
  EXPECT_EQ(client.try_submit(*request_message), ClientResult::kSuccess);

  auto result = std::make_unique<JobResultMessage>();
  ClientResult received{ClientResult::kEmpty};
  const auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (received == ClientResult::kEmpty && std::chrono::steady_clock::now() < timeout) {
    const JobServiceResult pumped = service.poll();
    ASSERT_NE(pumped, JobServiceResult::kTransportError);
    ASSERT_NE(pumped, JobServiceResult::kCorruptInput);
    received = client.try_receive(*result);
    if (received == ClientResult::kEmpty)
      std::this_thread::yield();
  }
  ASSERT_EQ(received, ClientResult::kSuccess);
  EXPECT_EQ(result->envelope.sequence, 500u);
  EXPECT_EQ(job_result_code(*result), JobResultCode::kComplete);

  JobDescriptor missing = descriptor;
  ++missing.state_schema;
  for (std::uint64_t sequence{501u}; sequence <= 503u; ++sequence) {
    ASSERT_EQ(make_job_request(*request_message, sequence, missing, 0u, input),
              ProtocolResult::kSuccess);
    ASSERT_EQ(client.try_submit(*request_message), ClientResult::kSuccess);
    const JobServiceResult pumped = service.poll();
    EXPECT_TRUE(pumped == JobServiceResult::kProgress ||
                pumped == JobServiceResult::kBackpressured);
  }
  EXPECT_EQ(service.poll(), JobServiceResult::kBackpressured);
  EXPECT_GT(service.stats().publish_blocked, 0u);

  std::array<bool, 3> rejected{};
  std::size_t rejected_count{};
  const auto rejection_timeout = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (rejected_count < rejected.size() && std::chrono::steady_clock::now() < rejection_timeout) {
    received = client.try_receive(*result);
    if (received == ClientResult::kSuccess) {
      ASSERT_GE(result->envelope.sequence, 501u);
      ASSERT_LE(result->envelope.sequence, 503u);
      const std::size_t index = static_cast<std::size_t>(result->envelope.sequence - 501u);
      EXPECT_FALSE(rejected[index]);
      rejected[index] = true;
      ++rejected_count;
      EXPECT_EQ(job_result_code(*result), JobResultCode::kAdapterNotFound);
    } else {
      ASSERT_EQ(received, ClientResult::kEmpty);
    }
    const JobServiceResult pumped = service.poll();
    ASSERT_NE(pumped, JobServiceResult::kTransportError);
    ASSERT_NE(pumped, JobServiceResult::kCorruptInput);
  }
  EXPECT_EQ(rejected_count, rejected.size());
  EXPECT_EQ(service.stats().requests_received, 4u);
  EXPECT_EQ(service.stats().requests_accepted, 1u);
  EXPECT_EQ(service.stats().requests_rejected, 3u);
  EXPECT_EQ(service.stats().results_published, 4u);

  request_message->envelope.magic = 0u;
  EXPECT_EQ(client.try_submit(*request_message), ClientResult::kInvalidRequest);
  EXPECT_EQ(client.try_shutdown(504u, descriptor.session_id), ClientResult::kSuccess);
  EXPECT_EQ(service.poll(), JobServiceResult::kStopped);
  EXPECT_TRUE(service.stopped());
}

TEST(JobEvents, BuffersValidatedLifecycleRecordsWithoutGrowth)
{
  JobDescriptor descriptor = test_job_descriptor(9u);
  descriptor.kind = JobKind::kStreaming;
  descriptor.deadline_ns = 0u;
  descriptor.total_work_units = 4u;
  JobEventMessage admitted{};
  ASSERT_EQ(make_job_event(admitted, 301u,
                           JobEventDetails{.event = JobEventKind::kAdmitted,
                                           .descriptor = descriptor,
                                           .progress = JobProgress{.state = JobState::kReady,
                                                                   .completed_work_units = 0u,
                                                                   .remaining_work_units = 4u},
                                           .timestamp_ns = 1'000u,
                                           .queue_depth = 1u}),
            ProtocolResult::kSuccess);
  EXPECT_EQ(validate(admitted), ProtocolResult::kSuccess);

  auto created = JobEventBuffer::create(1uz);
  ASSERT_TRUE(created) << created.error();
  EXPECT_EQ(created->try_push(admitted), JobEventBufferResult::kSuccess);
  EXPECT_EQ(created->try_push(admitted), JobEventBufferResult::kFull);
  EXPECT_EQ(created->size(), 1uz);
  EXPECT_EQ(created->dropped(), 1u);
  ASSERT_NE(created->front(), nullptr);
  EXPECT_EQ(created->front()->metadata.event, JobEventKind::kAdmitted);
  created->pop();
  EXPECT_TRUE(created->empty());

  admitted.metadata.reserved = 1u;
  EXPECT_EQ(created->try_push(admitted), JobEventBufferResult::kInvalid);
  EXPECT_EQ(created->dropped(), 1u);
}

TEST(JobMetrics, RecordsKindsProgressPreemptionMigrationAndLatency)
{
  JobDescriptor descriptor = test_job_descriptor(10u);
  descriptor.kind = JobKind::kStreaming;
  descriptor.deadline_ns = 0u;
  descriptor.total_work_units = 4u;
  JobMetrics metrics{};
  const auto record = [&](const JobEventDetails& details) {
    JobEventMessage event{};
    EXPECT_EQ(make_job_event(event, 401u, details), ProtocolResult::kSuccess);
    metrics.record(event);
  };
  record(JobEventDetails{.event = JobEventKind::kAdmitted,
                         .descriptor = descriptor,
                         .progress = JobProgress{.state = JobState::kReady,
                                                 .completed_work_units = 0u,
                                                 .remaining_work_units = 4u},
                         .timestamp_ns = 1'000u,
                         .queue_depth = 3u});
  record(JobEventDetails{.event = JobEventKind::kPreempted,
                         .descriptor = descriptor,
                         .progress = JobProgress{.state = JobState::kRunning,
                                                 .completed_work_units = 1u,
                                                 .remaining_work_units = 3u},
                         .timestamp_ns = 1'020u,
                         .related_generation = descriptor.generation + 1u,
                         .worker_index = 0u});
  record(JobEventDetails{.event = JobEventKind::kMigrationStarted,
                         .descriptor = descriptor,
                         .progress = JobProgress{.state = JobState::kRunning,
                                                 .completed_work_units = 1u,
                                                 .remaining_work_units = 3u},
                         .timestamp_ns = 1'030u,
                         .worker_index = 0u,
                         .peer_worker_index = 1u});
  record(JobEventDetails{.event = JobEventKind::kMigrationCompleted,
                         .descriptor = descriptor,
                         .progress = JobProgress{.state = JobState::kRunning,
                                                 .completed_work_units = 1u,
                                                 .remaining_work_units = 3u},
                         .timestamp_ns = 1'040u,
                         .worker_index = 1u,
                         .peer_worker_index = 0u});
  record(JobEventDetails{.event = JobEventKind::kCompleted,
                         .descriptor = descriptor,
                         .progress = JobProgress{.state = JobState::kComplete,
                                                 .completed_work_units = 4u,
                                                 .remaining_work_units = 0u},
                         .timestamp_ns = 1'100u,
                         .timing = JobEventTiming{.queue_ns = 10u,
                                                  .execution_ns = 80u,
                                                  .end_to_end_ns = 100u,
                                                  .cancellation_ns = 70u},
                         .worker_index = 1u,
                         .result_code = JobResultCode::kComplete});
  metrics.observe_workers(2uz, 1u);
  metrics.record_event_drops(2u);

  EXPECT_EQ(metrics.counters().events, 5u);
  EXPECT_EQ(metrics.counters().preempted, 1u);
  EXPECT_EQ(metrics.counters().migrations_completed, 1u);
  EXPECT_EQ(metrics.counters().completed_work_units, 4u);
  EXPECT_EQ(metrics.counters().queue_high_watermark, 3u);
  EXPECT_EQ(metrics.counters().busy_workers_high_watermark, 2u);
  EXPECT_EQ(metrics.counters().events_dropped, 2u);
  ASSERT_NE(metrics.counters(JobKind::kStreaming), nullptr);
  EXPECT_EQ(metrics.counters(JobKind::kStreaming)->completed, 1u);
  EXPECT_EQ(metrics.queue_latency().sum_ns(), 10u);
  EXPECT_EQ(metrics.execution_latency().sum_ns(), 80u);
  EXPECT_EQ(metrics.end_to_end_latency().sum_ns(), 100u);
  EXPECT_EQ(metrics.cancellation_latency().sum_ns(), 70u);

  std::ostringstream prometheus{};
  write_prometheus(prometheus, metrics);
  EXPECT_NE(prometheus.str().find(
                "flowedge_relay_job_completed_by_kind_total{kind=\"streaming\"} 1"),
            std::string::npos);
  std::ostringstream json{};
  write_metrics_json(json, metrics);
  EXPECT_NE(json.str().find("\"streaming\":{\"completed\":1"), std::string::npos);
  std::ostringstream otlp{};
  write_otlp_json(otlp, metrics);
  EXPECT_NE(otlp.str().find("\"job.kind\""), std::string::npos);
  EXPECT_NE(otlp.str().find("flowedge.relay.job.execution_latency"), std::string::npos);
}

TEST(CooperativeJob, MigratesIterativeStateBitExactly)
{
  TestCooperativeBackend source_backend{};
  auto source_result = make_iterative_job(source_backend, test_job_descriptor());
  ASSERT_TRUE(source_result) << source_result.error().message;
  CooperativeJob source = *source_result;
  ASSERT_TRUE(source.start());
  const auto partial = source.advance(2uz);
  ASSERT_TRUE(partial) << partial.error().message;
  EXPECT_EQ(partial->state, JobState::kRunning);
  EXPECT_EQ(partial->completed_work_units, 2u);

  std::vector<std::byte> capsule(source.capsule_bytes());
  const auto exported = source.export_capsule(capsule);
  ASSERT_TRUE(exported) << exported.error().message;
  EXPECT_EQ(*exported, capsule.size());
  const auto view = read_state_capsule(capsule);
  ASSERT_TRUE(view) << view.error().message;
  EXPECT_EQ(view->metadata.descriptor.kind, JobKind::kIterative);
  EXPECT_EQ(view->metadata.completed_work_units, 2u);

  TestCooperativeBackend migrated_backend{};
  auto migrated_result = make_iterative_job(migrated_backend, test_job_descriptor());
  ASSERT_TRUE(migrated_result) << migrated_result.error().message;
  CooperativeJob migrated = *migrated_result;
  const auto restored = migrated.restore_capsule(capsule);
  ASSERT_TRUE(restored) << restored.error().message;
  EXPECT_EQ(restored->state, JobState::kRunning);

  ASSERT_TRUE(source.advance(4uz));
  const auto migrated_complete = migrated.advance(8uz);
  ASSERT_TRUE(migrated_complete) << migrated_complete.error().message;
  EXPECT_EQ(migrated_complete->state, JobState::kComplete);
  EXPECT_EQ(source_backend.step, migrated_backend.step);
  EXPECT_EQ(source_backend.values, migrated_backend.values);
}

TEST(MambaStreamAdapter, MigratesExactCoreStateAndMatchesUninterruptedExecution)
{
  const std::filesystem::path model =
      std::filesystem::path{FLOWEDGE_SOURCE_DIR} / "models" / "mamba_flow.safetensors";
  if (!std::filesystem::exists(model))
    GTEST_SKIP() << "models/mamba_flow.safetensors is not available";

  constexpr std::array<std::int32_t, 4> tokens{1, 2, 3, 4};
  std::unique_ptr<fe_weights, decltype(&fe_weights_free)> weights{fe_weights_load(
                                                                      model.string().c_str()),
                                                                  fe_weights_free};
  ASSERT_NE(weights, nullptr) << fe_engine_last_error();
  auto opened_source = MambaStreamAdapter::open(weights.get(), tokens.size(), 0u);
  auto opened_destination = MambaStreamAdapter::open(weights.get(), tokens.size(), 0u);
  auto opened_reference = MambaStreamAdapter::open(weights.get(), tokens.size(), 0u);
  ASSERT_TRUE(opened_source) << opened_source.error();
  ASSERT_TRUE(opened_destination) << opened_destination.error();
  ASSERT_TRUE(opened_reference) << opened_reference.error();
  MambaStreamAdapter source = std::move(*opened_source);
  MambaStreamAdapter destination = std::move(*opened_destination);
  MambaStreamAdapter reference = std::move(*opened_reference);
  weights.reset();

  std::array<std::byte, tokens.size() * sizeof(std::int32_t)> request_bytes{};
  const auto encoded = encode_mamba_stream_request(tokens, request_bytes);
  ASSERT_TRUE(encoded) << encoded.error();
  const auto request = std::span{request_bytes}.first(*encoded);
  const JobDescriptor descriptor = source.make_descriptor(41u, 3u, tokens.size());
  ASSERT_TRUE(source.prepare(request));
  auto source_job_result = make_streaming_job(source, descriptor);
  ASSERT_TRUE(source_job_result) << source_job_result.error().message;
  CooperativeJob source_job = *source_job_result;
  ASSERT_TRUE(source_job.start());
  const auto partial = source_job.advance(2uz);
  ASSERT_TRUE(partial) << partial.error().message;
  EXPECT_EQ(partial->completed_work_units, 2u);

  std::vector<std::byte> capsule(source_job.capsule_bytes());
  const auto exported = source_job.export_capsule(capsule);
  ASSERT_TRUE(exported) << exported.error().message;
  EXPECT_EQ(*exported, capsule.size());

  auto destination_job_result = make_streaming_job(destination, descriptor);
  ASSERT_TRUE(destination_job_result) << destination_job_result.error().message;
  CooperativeJob destination_job = *destination_job_result;
  const auto restored = destination_job.restore_capsule(capsule);
  ASSERT_TRUE(restored) << restored.error().message;
  EXPECT_EQ(restored->completed_work_units, 2u);
  const auto migrated_complete = destination_job.advance(2uz);
  ASSERT_TRUE(migrated_complete) << migrated_complete.error().message;
  EXPECT_EQ(migrated_complete->state, JobState::kComplete);

  ASSERT_TRUE(reference.prepare(request));
  auto reference_job_result = make_streaming_job(reference, descriptor);
  ASSERT_TRUE(reference_job_result) << reference_job_result.error().message;
  CooperativeJob reference_job = *reference_job_result;
  ASSERT_TRUE(reference_job.start());
  const auto reference_complete = reference_job.advance(tokens.size());
  ASSERT_TRUE(reference_complete) << reference_complete.error().message;
  ASSERT_EQ(reference_complete->state, JobState::kComplete);
  ASSERT_EQ(destination.result().size(), reference.result().size());
  EXPECT_EQ(std::memcmp(destination.result().data(), reference.result().data(),
                        reference.result().size()),
            0);
}

TEST(MambaStreamAdapter, RoutesRealModelThroughWorkerPool)
{
  const std::filesystem::path model =
      std::filesystem::path{FLOWEDGE_SOURCE_DIR} / "models" / "mamba_flow.safetensors";
  if (!std::filesystem::exists(model))
    GTEST_SKIP() << "models/mamba_flow.safetensors is not available";

  constexpr std::array<std::int32_t, 4> tokens{1, 2, 3, 4};
  auto opened = MambaStreamAdapter::open(model.string(), tokens.size(), 0u);
  auto opened_reference = MambaStreamAdapter::open(model.string(), tokens.size(), 0u);
  ASSERT_TRUE(opened) << opened.error();
  ASSERT_TRUE(opened_reference) << opened_reference.error();
  MambaStreamAdapter adapter = std::move(*opened);
  MambaStreamAdapter reference = std::move(*opened_reference);
  std::array<std::byte, tokens.size() * sizeof(std::int32_t)> request_bytes{};
  const auto encoded = encode_mamba_stream_request(tokens, request_bytes);
  ASSERT_TRUE(encoded) << encoded.error();
  const auto payload = std::span{request_bytes}.first(*encoded);
  const JobDescriptor descriptor = adapter.make_descriptor(42u, 1u, tokens.size());

  ASSERT_TRUE(reference.prepare(payload));
  auto reference_job_result = make_streaming_job(reference, descriptor);
  ASSERT_TRUE(reference_job_result) << reference_job_result.error().message;
  CooperativeJob reference_job = *reference_job_result;
  ASSERT_TRUE(reference_job.start());
  ASSERT_TRUE(reference_job.advance(tokens.size()));

  std::array<JobAdapterRegistration, 1> entries{};
  std::array<JobAdapterRegistry, 1> registries{JobAdapterRegistry{entries}};
  ASSERT_TRUE(registries[0].add(adapter.registration()));
  registries[0].freeze();
  auto created = JobWorkerPool::create(registries, 2uz, {}, 2uz, 1uz);
  ASSERT_TRUE(created) << created.error();
  JobWorkerPool pool = std::move(*created);
  auto request_message = std::make_unique<JobRequestMessage>();
  ASSERT_EQ(make_job_request(*request_message, 501u, descriptor, 0u, payload),
            ProtocolResult::kSuccess);
  ASSERT_EQ(pool.submit(*request_message), JobSubmitResult::kAccepted);

  const JobResultMessage* result{};
  const auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while (result == nullptr && std::chrono::steady_clock::now() < timeout) {
    result = pool.ready_result();
    if (result == nullptr)
      std::this_thread::yield();
  }
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(validate(*result), ProtocolResult::kSuccess);
  EXPECT_EQ(job_result_code(*result), JobResultCode::kComplete);
  EXPECT_EQ(result->metadata.progress.completed_work_units, tokens.size());
  ASSERT_EQ(result->payload_values().size(), reference.result().size());
  EXPECT_EQ(std::memcmp(result->payload_values().data(), reference.result().data(),
                        reference.result().size()),
            0);
  pool.release_ready_result();
  EXPECT_TRUE(pool.release_session(descriptor.session_id));
}

TEST(CooperativeJob, RejectsCorruptAndIncompatibleCapsulesBeforeRestore)
{
  TestCooperativeBackend backend{};
  auto source_result = make_streaming_job(backend, test_job_descriptor());
  ASSERT_TRUE(source_result);
  CooperativeJob source = *source_result;
  ASSERT_TRUE(source.start());
  ASSERT_TRUE(source.advance(1uz));
  std::vector<std::byte> capsule(source.capsule_bytes());
  ASSERT_TRUE(source.export_capsule(capsule));

  std::vector<std::byte> corrupt = capsule;
  corrupt[kStateCapsuleHeaderBytes] ^= std::byte{0x40};
  TestCooperativeBackend corrupt_destination{};
  auto corrupt_result = make_streaming_job(corrupt_destination, test_job_descriptor());
  ASSERT_TRUE(corrupt_result);
  CooperativeJob corrupt_job = *corrupt_result;
  const auto corrupt_restore = corrupt_job.restore_capsule(corrupt);
  ASSERT_FALSE(corrupt_restore);
  EXPECT_EQ(corrupt_restore.error().code, JobErrorCode::kCapsuleInvalid);
  EXPECT_EQ(corrupt_job.progress().state, JobState::kReady);
  EXPECT_TRUE(corrupt_job.restore_capsule(capsule));

  JobDescriptor incompatible_descriptor = test_job_descriptor();
  ++incompatible_descriptor.state_schema;
  TestCooperativeBackend incompatible_backend{};
  auto incompatible_result = make_streaming_job(incompatible_backend, incompatible_descriptor);
  ASSERT_TRUE(incompatible_result);
  CooperativeJob incompatible = *incompatible_result;
  const auto incompatible_restore = incompatible.restore_capsule(capsule);
  ASSERT_FALSE(incompatible_restore);
  EXPECT_EQ(incompatible_restore.error().code, JobErrorCode::kCapsuleIncompatible);
  EXPECT_EQ(incompatible.progress().state, JobState::kReady);
}

TEST(CooperativeJob, ClassifiesAdaptersAndCancelsAtAWorkBoundary)
{
  TestCooperativeBackend iterative_backend{};
  TestCooperativeBackend streaming_backend{};
  TestCooperativeBackend speculative_backend{};
  auto iterative = make_iterative_job(iterative_backend, test_job_descriptor());
  auto streaming = make_streaming_job(streaming_backend, test_job_descriptor());
  auto speculative = make_speculative_job(speculative_backend, test_job_descriptor());
  ASSERT_TRUE(iterative);
  ASSERT_TRUE(streaming);
  ASSERT_TRUE(speculative);
  EXPECT_EQ(iterative->descriptor().kind, JobKind::kIterative);
  EXPECT_EQ(streaming->descriptor().kind, JobKind::kStreaming);
  EXPECT_EQ(speculative->descriptor().kind, JobKind::kSpeculative);

  ASSERT_TRUE(streaming->start());
  ASSERT_TRUE(streaming->advance(1uz));
  const auto cancelled = streaming->advance(1uz, streaming->descriptor().generation + 1u);
  ASSERT_TRUE(cancelled);
  EXPECT_EQ(cancelled->state, JobState::kCancelled);
  EXPECT_TRUE(streaming_backend.cancelled);
  EXPECT_EQ(cancelled->completed_work_units, 1u);
}

TEST(EdfScheduler, EnforcesFreshnessThenOrdersByDeadline)
{
  EdfScheduler scheduler{4uz};
  EXPECT_EQ(scheduler.submit(request(1u, 1u, 400u)), SubmitResult::kAccepted);
  EXPECT_EQ(scheduler.submit(request(2u, 1u, 300u)), SubmitResult::kAccepted);
  EXPECT_EQ(scheduler.submit(request(3u, 2u, 500u)), SubmitResult::kAccepted);
  EXPECT_EQ(scheduler.size(), 1uz);
  EXPECT_EQ(scheduler.submit(request(4u, 1u, 200u)), SubmitResult::kStale);
  EXPECT_EQ(scheduler.submit(request(5u, 2u, 250u)), SubmitResult::kAccepted);

  const ConditionMessage* first = scheduler.pop(100u);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first->envelope.sequence, 5u);
  const ConditionMessage* second = scheduler.pop(100u);
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(second->envelope.sequence, 3u);
  EXPECT_FALSE(scheduler.pop(100u));
  EXPECT_EQ(scheduler.stats().stale, 3u);
}

TEST(EdfScheduler, ExpiresRequestsAndEvictsLatestDeadline)
{
  EdfScheduler scheduler{2uz};
  EXPECT_EQ(scheduler.submit(request(1u, 4u, 400u)), SubmitResult::kAccepted);
  EXPECT_EQ(scheduler.submit(request(2u, 4u, 500u)), SubmitResult::kAccepted);
  ActionMessage displaced{};
  EXPECT_EQ(scheduler.submit(request(3u, 4u, 300u), {}, &displaced),
            SubmitResult::kAcceptedAndEvicted);
  EXPECT_EQ(validate(displaced), ProtocolResult::kSuccess);
  EXPECT_EQ(action_code(displaced), RelayActionCode::kRejectedCapacity);
  EXPECT_EQ(displaced.envelope.sequence, 2u);
  ASSERT_EQ(scheduler.stats().evicted, 1u);

  ActionMessage expired{};
  EXPECT_FALSE(scheduler.pop(350u, &expired));
  EXPECT_EQ(validate(expired), ProtocolResult::kSuccess);
  EXPECT_EQ(action_code(expired), RelayActionCode::kExpired);
  EXPECT_EQ(expired.envelope.sequence, 3u);

  const ConditionMessage* first = scheduler.pop(350u);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first->envelope.sequence, 1u);
  EXPECT_FALSE(scheduler.pop(600u));
  EXPECT_EQ(scheduler.stats().expired, 1u);
}

TEST(EdfScheduler, RejectsUnreachableDeadlinePrefixesWithoutPruningFeasibleWork)
{
  EdfScheduler scheduler{4uz, AdmissionPolicy{.nanoseconds_per_nfe = 10u, .reserve_ns = 5u}};
  const AdmissionContext idle{.now_ns = 100u};
  EXPECT_EQ(scheduler.submit(request(1u, 4u, 200u), idle), SubmitResult::kAccepted);
  EXPECT_EQ(scheduler.submit(request(2u, 4u, 250u), idle), SubmitResult::kDeadlineUnreachable);
  EXPECT_EQ(scheduler.size(), 1uz);
  EXPECT_EQ(scheduler.stats().unreachable, 1u);

  // A fresh generation cancels the older queued prefix, so it is evaluated on
  // its own and does not destroy feasible work unless admission succeeds.
  EXPECT_EQ(scheduler.submit(request(3u, 5u, 200u), idle), SubmitResult::kAccepted);
  EXPECT_EQ(scheduler.size(), 1uz);
  EXPECT_EQ(scheduler.newest_generation(), 5u);

  EdfScheduler active_scheduler{2uz, AdmissionPolicy{.nanoseconds_per_nfe = 10u, .reserve_ns = 5u}};
  AdmissionContext active{.now_ns = 100u};
  active.active[0] = {.generation = 4u, .remaining_nfe = 2u};
  EXPECT_EQ(active_scheduler.submit(request(4u, 4u, 200u), active),
            SubmitResult::kDeadlineUnreachable);

  EdfScheduler replacement{1uz, AdmissionPolicy{.nanoseconds_per_nfe = 10u, .reserve_ns = 0u}};
  EXPECT_EQ(replacement.submit(request(5u, 4u, 250u), idle), SubmitResult::kAccepted);
  ActionMessage replaced{};
  EXPECT_EQ(replacement.submit(request(6u, 4u, 200u), idle, &replaced),
            SubmitResult::kAcceptedAndEvicted);
  EXPECT_EQ(replaced.envelope.sequence, 5u);

  EdfScheduler saturated{2uz,
                         AdmissionPolicy{.nanoseconds_per_nfe =
                                             std::numeric_limits<std::uint64_t>::max(),
                                         .reserve_ns = std::numeric_limits<std::uint64_t>::max()}};
  EXPECT_EQ(saturated.submit(request(7u, 4u, std::numeric_limits<std::uint64_t>::max()), idle),
            SubmitResult::kDeadlineUnreachable);
}

TEST(EdfScheduler, AdmitsIndependentDeadlineWorkAcrossWorkerLanes)
{
  const AdmissionPolicy policy{.nanoseconds_per_nfe = 10u, .reserve_ns = 0u};
  EdfScheduler single_lane{4uz, policy};
  EdfScheduler two_lanes{4uz, policy};
  const ConditionMessage first = request(20u, 8u, 190u); // 8 NFE = 80 ns
  const ConditionMessage second = request(21u, 8u, 190u);
  const AdmissionContext single{.now_ns = 100u, .worker_count = 1uz};
  const AdmissionContext parallel{.now_ns = 100u, .worker_count = 2uz};

  EXPECT_EQ(single_lane.submit(first, single), SubmitResult::kAccepted);
  EXPECT_EQ(single_lane.submit(second, single), SubmitResult::kDeadlineUnreachable);
  EXPECT_EQ(two_lanes.submit(first, parallel), SubmitResult::kAccepted);
  EXPECT_EQ(two_lanes.submit(second, parallel), SubmitResult::kAccepted);

  EdfScheduler occupied{4uz, policy};
  AdmissionContext one_busy{.now_ns = 100u, .worker_count = 2uz};
  one_busy.active[0] = {.generation = 8u, .remaining_nfe = 8u};
  EXPECT_EQ(occupied.submit(first, one_busy), SubmitResult::kAccepted);
  EXPECT_EQ(occupied.submit(second, one_busy), SubmitResult::kDeadlineUnreachable);
}

TEST(SharedMemoryRing, ExchangesChecksummedVariableSizedMessages)
{
  const std::string name = unique_name("flowedge-relay-ring-test");
  auto created = SharedMemoryRing::create(name, RingConfig{2u, sizeof(ConditionMessage)});
  ASSERT_TRUE(created) << created.error();
  auto opened = SharedMemoryRing::open(name);
  ASSERT_TRUE(opened) << opened.error();
  SharedMemoryRing producer = std::move(*created);
  SharedMemoryRing consumer = std::move(*opened);

  const auto message = std::make_unique<ConditionMessage>(request(42u, 8u, 1'000u));
  EXPECT_EQ(producer.try_push(wire_bytes(*message)), RingResult::kSuccess);
  EXPECT_EQ(producer.try_push(wire_bytes(*message)), RingResult::kSuccess);
  EXPECT_EQ(producer.try_push(wire_bytes(*message)), RingResult::kFull);
  EXPECT_EQ(consumer.size(), 2u);

  auto received = std::make_unique<ConditionMessage>();
  std::size_t bytes{};
  EXPECT_EQ(consumer.try_pop(std::as_writable_bytes(std::span{received.get(), 1uz}), bytes),
            RingResult::kSuccess);
  EXPECT_EQ(bytes, wire_size(*received));
  EXPECT_EQ(received->envelope.sequence, 42u);
  EXPECT_EQ(received->metadata.generation, 8u);
  EXPECT_FLOAT_EQ(received->condition_values()[2], 3.0f);

  const ControlMessage shutdown = make_shutdown_message(43u, 77u);
  EXPECT_EQ(producer.try_push(shutdown), RingResult::kSuccess);
  EXPECT_EQ(consumer.try_pop(std::as_writable_bytes(std::span{received.get(), 1uz}), bytes),
            RingResult::kSuccess);
  EXPECT_EQ(consumer.try_pop(std::as_writable_bytes(std::span{received.get(), 1uz}), bytes),
            RingResult::kSuccess);
  EXPECT_EQ(bytes, sizeof(ControlMessage));
  EXPECT_EQ(received->envelope.kind, MessageKind::kShutdown);
  EXPECT_EQ(consumer.try_pop(std::as_writable_bytes(std::span{received.get(), 1uz}), bytes),
            RingResult::kEmpty);
}

TEST(RelayTrace, ReplaysExactConditionAndActionRecords)
{
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / (unique_name("flowedge-relay") + ".trace");
  const ConditionMessage condition = request(7u, 11u, 900u);
  ActionMessage action{};
  action.envelope.sequence = condition.envelope.sequence;
  action.envelope.session_id = condition.envelope.session_id;
  action.metadata.struct_size = sizeof(action.metadata);
  action.metadata.protocol_version = FE_PROTOCOL_VERSION;
  action.metadata.status = FE_ACTION_COMPLETE;
  action.metadata.generation = condition.metadata.generation;
  action.metadata.condition_dim = condition.metadata.condition_dim;
  action.metadata.action_dim = condition.metadata.action_dim;
  action.envelope.struct_size = static_cast<std::uint32_t>(wire_size(action));
  action.action[0] = 0.5f;

  JobDescriptor descriptor = test_job_descriptor(11u);
  descriptor.kind = JobKind::kIterative;
  descriptor.deadline_ns = 0u;
  descriptor.total_work_units = 2u;
  std::array<std::byte, sizeof(std::uint64_t)> job_payload{};
  StateWriter payload_writer{job_payload};
  ASSERT_TRUE(payload_writer.write(std::uint64_t{42u}));
  auto job_request = std::make_unique<JobRequestMessage>();
  ASSERT_EQ(make_job_request(*job_request, 80u, descriptor, 1'000u, job_payload),
            ProtocolResult::kSuccess);
  auto job_result = std::make_unique<JobResultMessage>();
  ASSERT_EQ(make_job_result(*job_result, 80u, descriptor,
                            JobProgress{.state = JobState::kComplete,
                                        .completed_work_units = 2u,
                                        .remaining_work_units = 0u},
                            1'100u, JobResultCode::kComplete, job_payload),
            ProtocolResult::kSuccess);
  JobEventMessage job_event{};
  ASSERT_EQ(make_job_event(job_event, 80u,
                           JobEventDetails{.event = JobEventKind::kCompleted,
                                           .descriptor = descriptor,
                                           .progress = JobProgress{.state = JobState::kComplete,
                                                                   .completed_work_units = 2u,
                                                                   .remaining_work_units = 0u},
                                           .timestamp_ns = 1'100u,
                                           .timing = JobEventTiming{.queue_ns = 10u,
                                                                    .execution_ns = 80u,
                                                                    .end_to_end_ns = 100u},
                                           .worker_index = 1u,
                                           .result_code = JobResultCode::kComplete}),
            ProtocolResult::kSuccess);

  {
    auto opened = TraceWriter::open(path.string().c_str());
    ASSERT_TRUE(opened) << opened.error();
    TraceWriter writer = std::move(*opened);
    ASSERT_TRUE(writer.append(condition));
    ASSERT_TRUE(writer.append(action));
    ASSERT_TRUE(writer.append(*job_request));
    ASSERT_TRUE(writer.append(*job_result));
    ASSERT_TRUE(writer.append(job_event));
    writer.flush();
  }

  // Trace v2 is canonical little-endian rather than a dump of native structs.
  // Check the file and record magics plus representative integer/float bytes.
  {
    std::ifstream file{path, std::ios::binary};
    ASSERT_TRUE(file);
    std::array<unsigned char, 172uz> bytes{};
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    ASSERT_EQ(file.gcount(), static_cast<std::streamsize>(bytes.size()));
    EXPECT_EQ(std::memcmp(bytes.data(), "FETRACE2", 8uz), 0);
    EXPECT_EQ(bytes[8], 2u);
    EXPECT_EQ(bytes[9], 0u);
    EXPECT_EQ(std::memcmp(bytes.data() + 16uz, "REC2", 4uz), 0);
    EXPECT_EQ(std::memcmp(bytes.data() + 48uz, "FRE1", 4uz), 0);
    EXPECT_EQ(bytes[64], 7u);  // envelope.sequence, little-endian
    EXPECT_EQ(bytes[72], 77u); // envelope.session_id, little-endian
    EXPECT_EQ(bytes[168], 0u); // first condition float: 1.0f = 0x3f800000
    EXPECT_EQ(bytes[169], 0u);
    EXPECT_EQ(bytes[170], 0x80u);
    EXPECT_EQ(bytes[171], 0x3fu);
  }
  auto opened = TraceReader::open(path.string().c_str());
  ASSERT_TRUE(opened) << opened.error();
  TraceReader reader = std::move(*opened);
  const auto first = reader.next();
  ASSERT_TRUE(first);
  ASSERT_TRUE(*first);
  EXPECT_EQ((*first)->kind, MessageKind::kCondition);
  EXPECT_EQ((*first)->sequence, 7u);
  ASSERT_EQ((*first)->message.size(), wire_size(condition));
  EXPECT_EQ(std::memcmp((*first)->message.data(), &condition, wire_size(condition)), 0);

  const auto second = reader.next();
  ASSERT_TRUE(second);
  ASSERT_TRUE(*second);
  EXPECT_EQ((*second)->kind, MessageKind::kAction);
  EXPECT_EQ((*second)->sequence, 7u);
  ASSERT_EQ((*second)->message.size(), wire_size(action));
  EXPECT_EQ(std::memcmp((*second)->message.data(), &action, wire_size(action)), 0);
  const auto third = reader.next();
  ASSERT_TRUE(third);
  ASSERT_TRUE(*third);
  EXPECT_EQ((*third)->kind, MessageKind::kJobRequest);
  ASSERT_EQ((*third)->message.size(), wire_size(*job_request));
  EXPECT_EQ(std::memcmp((*third)->message.data(), job_request.get(), wire_size(*job_request)), 0);

  const auto fourth = reader.next();
  ASSERT_TRUE(fourth);
  ASSERT_TRUE(*fourth);
  EXPECT_EQ((*fourth)->kind, MessageKind::kJobResult);
  ASSERT_EQ((*fourth)->message.size(), wire_size(*job_result));
  EXPECT_EQ(std::memcmp((*fourth)->message.data(), job_result.get(), wire_size(*job_result)), 0);

  const auto fifth = reader.next();
  ASSERT_TRUE(fifth);
  ASSERT_TRUE(*fifth);
  EXPECT_EQ((*fifth)->kind, MessageKind::kJobEvent);
  ASSERT_EQ((*fifth)->message.size(), sizeof(job_event));
  EXPECT_EQ(std::memcmp((*fifth)->message.data(), &job_event, sizeof(job_event)), 0);
  const auto eof = reader.next();
  ASSERT_TRUE(eof);
  EXPECT_FALSE(*eof);
  std::error_code ignored{};
  std::filesystem::remove(path, ignored);
}

TEST(RelayMetrics, RecordsTypedOutcomesAndFixedLatencyHistograms)
{
  RelayMetrics metrics{};
  metrics.record_condition(true);
  metrics.record_condition(false);
  metrics.record_accepted(3uz);

  ActionMessage complete{};
  complete.envelope.flags = static_cast<std::uint32_t>(RelayActionCode::kInference);
  complete.metadata.status = FE_ACTION_COMPLETE;
  complete.metadata.timestamp_ns = 100u;
  metrics.record_action(complete, 170u,
                        WorkerTiming{.dispatched_ns = 110u,
                                     .started_ns = 120u,
                                     .finished_ns = 150u});
  const ConditionMessage condition = request(90u, 11u, 300u);
  ActionMessage rejected{};
  make_rejected_action(rejected, condition, 200u, RelayActionCode::kRejectedDeadline);
  metrics.record_action(rejected, 200u);
  metrics.record_action_drop();
  metrics.observe_workers(2uz, 1u);

  const RelayMetricCounters& counters = metrics.counters();
  EXPECT_EQ(counters.conditions_received, 2u);
  EXPECT_EQ(counters.conditions_corrupt, 1u);
  EXPECT_EQ(counters.conditions_accepted, 1u);
  EXPECT_EQ(counters.actions_published, 2u);
  EXPECT_EQ(counters.actions_dropped, 1u);
  EXPECT_EQ(counters.inference_complete, 1u);
  EXPECT_EQ(counters.rejected_deadline, 1u);
  EXPECT_EQ(counters.queue_high_watermark, 3u);
  EXPECT_EQ(counters.busy_workers_high_watermark, 2u);
  EXPECT_EQ(counters.worker_failures, 1u);
  EXPECT_EQ(metrics.queue_latency().count(), 1u);
  EXPECT_EQ(metrics.queue_latency().sum_ns(), 20u);
  EXPECT_EQ(metrics.execution_latency().sum_ns(), 30u);
  EXPECT_EQ(metrics.end_to_end_latency().count(), 2u);

  std::ostringstream prometheus{};
  write_prometheus(prometheus, metrics);
  EXPECT_NE(prometheus.str().find("flowedge_relay_inference_complete_total 1"), std::string::npos);
  EXPECT_NE(prometheus.str().find("flowedge_relay_execution_latency_ns_bucket"), std::string::npos);
  std::ostringstream json{};
  write_metrics_json(json, metrics);
  EXPECT_NE(json.str().find("\"rejected_deadline\":1"), std::string::npos);
  std::ostringstream otlp{};
  write_otlp_json(otlp, metrics);
  EXPECT_NE(otlp.str().find("\"resourceMetrics\""), std::string::npos);
  EXPECT_NE(otlp.str().find("flowedge.relay.execution_latency"), std::string::npos);
}

TEST(HeadWorker, RunsAndCancelsGenerationTrackedRequests)
{
  const std::filesystem::path model =
      std::filesystem::path{FLOWEDGE_SOURCE_DIR} / "models" / "mamba_flow.safetensors";
  if (!std::filesystem::exists(model))
    GTEST_SKIP() << "models/mamba_flow.safetensors is not available";
  auto opened = HeadWorker::open(model.string(), 0u);
  ASSERT_TRUE(opened) << opened.error();
  HeadWorker worker = std::move(*opened);
  const auto& metadata = worker.model_metadata();
  std::vector<float> condition(metadata.condition_dim, 0.125f);
  std::vector<float> noise(metadata.action_dim, -0.25f);

  ConditionMessage message{};
  ASSERT_EQ(make_condition_message(message, metadata, 1u, 9u, 100u, 0u, 5u, 2uz, FE_SOLVER_EULER,
                                   condition, noise),
            ProtocolResult::kSuccess);
  ASSERT_TRUE(worker.begin(message)) << worker.last_error();
  ActionMessage action{};
  WorkerStep step{};
  do {
    step = worker.advance(action);
  } while (step == WorkerStep::kInProgress);
  EXPECT_EQ(step, WorkerStep::kComplete) << worker.last_error();
  EXPECT_EQ(validate(action), ProtocolResult::kSuccess);
  EXPECT_EQ(action.metadata.status, FE_ACTION_COMPLETE);
  EXPECT_EQ(action.metadata.generation, 5u);

  message.envelope.sequence = 2u;
  message.metadata.generation = 6u;
  ASSERT_TRUE(worker.begin(message)) << worker.last_error();
  worker.cancel_before(7u);
  EXPECT_EQ(worker.advance(action), WorkerStep::kCancelled);
  EXPECT_EQ(action.metadata.status, FE_ACTION_CANCELLED);
  EXPECT_EQ(action.metadata.generation, 6u);
}

TEST(CoreWeights, SharesImmutableCheckpointAcrossIndependentEngineState)
{
  const std::filesystem::path model =
      std::filesystem::path{FLOWEDGE_SOURCE_DIR} / "models" / "mamba_flow.safetensors";
  if (!std::filesystem::exists(model))
    GTEST_SKIP() << "models/mamba_flow.safetensors is not available";

  std::unique_ptr<fe_weights, decltype(&fe_weights_free)> weights{fe_weights_load(
                                                                      model.string().c_str()),
                                                                  fe_weights_free};
  ASSERT_NE(weights, nullptr) << fe_engine_last_error();
  EXPECT_GT(fe_weights_size_bytes(weights.get()), 0uz);
  std::unique_ptr<fe_engine, decltype(&fe_engine_free)> first{
      fe_engine_create_from_weights(weights.get(), 0u), fe_engine_free};
  std::unique_ptr<fe_engine, decltype(&fe_engine_free)> second{
      fe_engine_create_from_weights(weights.get(), 0u), fe_engine_free};
  ASSERT_NE(first, nullptr) << fe_engine_last_error();
  ASSERT_NE(second, nullptr) << fe_engine_last_error();
  weights.reset(); // engines retain the shared immutable checkpoint lifetime

  fe_model_metadata first_metadata{};
  fe_model_metadata second_metadata{};
  ASSERT_EQ(fe_engine_model_metadata(first.get(), &first_metadata), 0);
  ASSERT_EQ(fe_engine_model_metadata(second.get(), &second_metadata), 0);
  EXPECT_TRUE(
      std::ranges::equal(first_metadata.model_digest.bytes, second_metadata.model_digest.bytes));
  std::vector<float> condition(first_metadata.condition_dim, 0.125f);
  std::vector<float> noise(first_metadata.action_dim, -0.25f);
  std::vector<float> first_action(first_metadata.action_dim);
  std::vector<float> second_action(first_metadata.action_dim);
  ASSERT_EQ(fe_engine_sample_condition(first.get(), condition.data(), noise.data(), 6uz,
                                       FE_SOLVER_HEUN, first_action.data()),
            0);
  ASSERT_EQ(fe_engine_sample_condition(second.get(), condition.data(), noise.data(), 6uz,
                                       FE_SOLVER_HEUN, second_action.data()),
            0);
  EXPECT_EQ(first_action, second_action);
}

TEST(WorkerTopology, ParsesPortablePlacementPolicies)
{
  const auto none = parse_worker_placement("none");
  const auto compact = parse_worker_placement("compact");
  const auto spread = parse_worker_placement("spread");
  ASSERT_TRUE(none);
  ASSERT_TRUE(compact);
  ASSERT_TRUE(spread);
  EXPECT_EQ(*none, WorkerPlacement::kNone);
  EXPECT_EQ(*compact, WorkerPlacement::kCompact);
  EXPECT_EQ(*spread, WorkerPlacement::kSpread);
  EXPECT_FALSE(parse_worker_placement("random"));
}

TEST(HeadWorkerPool, RunsTwoRequestsOnPreallocatedWorkerThreads)
{
  const std::filesystem::path model =
      std::filesystem::path{FLOWEDGE_SOURCE_DIR} / "models" / "mamba_flow.safetensors";
  if (!std::filesystem::exists(model))
    GTEST_SKIP() << "models/mamba_flow.safetensors is not available";
  auto opened = HeadWorkerPool::open(model.string(), 2uz, 0u, WorkerPlacement::kSpread);
  ASSERT_TRUE(opened) << opened.error();
  HeadWorkerPool pool = std::move(*opened);
  EXPECT_GT(pool.shared_weight_bytes(), 0uz);
  const auto& metadata = pool.model_metadata();
  std::vector<float> condition(metadata.condition_dim, 0.125f);
  std::vector<float> noise(metadata.action_dim, -0.25f);

  ConditionMessage first{};
  ConditionMessage second{};
  ASSERT_EQ(make_condition_message(first, metadata, 101u, 9u, 100u, 0u, 7u, 20uz, FE_SOLVER_HEUN,
                                   condition, noise),
            ProtocolResult::kSuccess);
  ASSERT_EQ(make_condition_message(second, metadata, 102u, 9u, 100u, 0u, 7u, 20uz, FE_SOLVER_HEUN,
                                   condition, noise),
            ProtocolResult::kSuccess);
  ASSERT_TRUE(pool.try_dispatch(first)) << pool.last_error();
  ASSERT_TRUE(pool.try_dispatch(second)) << pool.last_error();
  EXPECT_FALSE(pool.try_dispatch(first));

  std::array<bool, 2> received{};
  std::size_t completed{};
  const auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while (completed < received.size() && std::chrono::steady_clock::now() < timeout) {
    const ActionMessage* const action = pool.ready_action();
    if (action == nullptr) {
      std::this_thread::yield();
      continue;
    }
    ASSERT_EQ(validate(*action), ProtocolResult::kSuccess);
    ASSERT_EQ(action->metadata.status, FE_ACTION_COMPLETE);
    ASSERT_GE(action->envelope.sequence, 101u);
    ASSERT_LE(action->envelope.sequence, 102u);
    const std::size_t index = static_cast<std::size_t>(action->envelope.sequence - 101u);
    EXPECT_FALSE(received[index]);
    received[index] = true;
    ++completed;
    pool.release_ready_action();
  }
  EXPECT_EQ(completed, received.size());
  EXPECT_EQ(pool.busy_count(), 0uz);
  EXPECT_TRUE(pool.has_idle());
}

TEST(HeadWorkerPool, CancelsDispatchedWorkAndInvalidatesBorrowedStaleResults)
{
  const std::filesystem::path model =
      std::filesystem::path{FLOWEDGE_SOURCE_DIR} / "models" / "mamba_flow.safetensors";
  if (!std::filesystem::exists(model))
    GTEST_SKIP() << "models/mamba_flow.safetensors is not available";
  auto opened = HeadWorkerPool::open(model.string(), 1uz, 0u);
  ASSERT_TRUE(opened) << opened.error();
  HeadWorkerPool pool = std::move(*opened);
  const auto& metadata = pool.model_metadata();
  std::vector<float> condition(metadata.condition_dim, 0.125f);
  std::vector<float> noise(metadata.action_dim, -0.25f);

  ConditionMessage request{};
  ASSERT_EQ(make_condition_message(request, metadata, 201u, 10u, 100u, 0u, 5u, 1'000uz,
                                   FE_SOLVER_HEUN, condition, noise),
            ProtocolResult::kSuccess);
  ASSERT_TRUE(pool.try_dispatch(request)) << pool.last_error();
  pool.cancel_before(6u);

  const ActionMessage* action{};
  const auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while (action == nullptr && std::chrono::steady_clock::now() < timeout) {
    action = pool.ready_action();
    if (action == nullptr)
      std::this_thread::yield();
  }
  ASSERT_NE(action, nullptr);
  ASSERT_EQ(validate(*action), ProtocolResult::kSuccess);
  EXPECT_EQ(action->metadata.status, FE_ACTION_CANCELLED);
  EXPECT_EQ(action->metadata.generation, 5u);

  pool.cancel_before(6u);
  EXPECT_EQ(pool.ready_action(), nullptr);
  EXPECT_TRUE(pool.has_idle());
}

} // namespace
