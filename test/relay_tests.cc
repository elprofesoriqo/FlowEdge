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
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
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

  ActionMessage rejected{};
  make_rejected_action(rejected, message, 101u, RelayActionCode::kRejectedDeadline);
  EXPECT_EQ(validate(rejected), ProtocolResult::kSuccess);
  EXPECT_TRUE(compatible(rejected, test_model()));
  EXPECT_EQ(action_code(rejected), RelayActionCode::kRejectedDeadline);
  EXPECT_EQ(rejected.metadata.status, FE_ACTION_FAILED);
  EXPECT_EQ(rejected.metadata.remaining_nfe, message.metadata.remaining_nfe);
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
  EXPECT_EQ(active_scheduler.submit(request(4u, 4u, 200u),
                                    AdmissionContext{.now_ns = 100u,
                                                     .active_generation = 4u,
                                                     .active_remaining_nfe = 2u}),
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
