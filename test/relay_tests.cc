#include "relay/protocol/messages.h"
#include "relay/protocol/trace.h"
#include "relay/scheduler/edf_scheduler.h"
#include "relay/shared_memory/shared_memory_ring.h"
#include "relay/worker/head_worker.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace fe::relay;

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
  EXPECT_EQ(scheduler.submit(request(3u, 4u, 300u)), SubmitResult::kAcceptedAndEvicted);
  ASSERT_EQ(scheduler.stats().evicted, 1u);

  const ConditionMessage* first = scheduler.pop(350u);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first->envelope.sequence, 1u);
  EXPECT_FALSE(scheduler.pop(600u));
  EXPECT_EQ(scheduler.stats().expired, 1u);
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

  {
    auto opened = TraceWriter::open(path.string().c_str());
    ASSERT_TRUE(opened) << opened.error();
    TraceWriter writer = std::move(*opened);
    ASSERT_TRUE(writer.append(condition));
    ASSERT_TRUE(writer.append(action));
    writer.flush();
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
  const auto eof = reader.next();
  ASSERT_TRUE(eof);
  EXPECT_FALSE(*eof);
  std::error_code ignored{};
  std::filesystem::remove(path, ignored);
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

} // namespace
