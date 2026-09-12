#include "relay_test_utils.h"

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
  ASSERT_EQ(make_job_request(*request_message, 91u, descriptor, 100u, input,
                             JobServiceClass::kInteractive),
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
  EXPECT_EQ(result_message->metadata.service_class, JobServiceClass::kInteractive);
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

  const std::string ring_name = unique_name("fe-gj-ring");
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

TEST(JobWorkerPool, EmitsAdmissionBeforeDispatchForFreshGeneration)
{
  JobDescriptor descriptor = test_job_descriptor(20u);
  descriptor.kind = JobKind::kIterative;
  descriptor.deadline_ns = 0u;
  descriptor.total_work_units = TestRoutedBackend::kTotalSteps;
  TestRoutedBackend backend{};
  std::array<JobAdapterRegistration, 1> entries{};
  std::array<JobAdapterRegistry, 1> registries{JobAdapterRegistry{entries}};
  ASSERT_TRUE(registries[0].add(make_routed_adapter(backend, job_route(descriptor),
                                                    sizeof(std::uint64_t), sizeof(std::uint64_t))));
  registries[0].freeze();
  auto events = JobEventBuffer::create(16uz);
  ASSERT_TRUE(events) << events.error();
  auto created =
      JobWorkerPool::create(registries, 2uz, {}, 2uz, 1uz, WorkerPlacement::kNone, &*events);
  ASSERT_TRUE(created) << created.error();
  JobWorkerPool pool = std::move(*created);

  std::array<std::byte, sizeof(std::uint64_t)> input{};
  StateWriter writer{input};
  ASSERT_TRUE(writer.write(std::uint64_t{2u}));
  auto request_message = std::make_unique<JobRequestMessage>();
  ASSERT_EQ(make_job_request(*request_message, 410u, descriptor, 0u, input),
            ProtocolResult::kSuccess);
  ASSERT_EQ(pool.submit(*request_message), JobSubmitResult::kAccepted);

  const JobResultMessage* completed{};
  const auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (completed == nullptr && std::chrono::steady_clock::now() < timeout) {
    completed = pool.ready_result();
    if (completed == nullptr)
      std::this_thread::yield();
  }
  ASSERT_NE(completed, nullptr);
  pool.release_ready_result();
  while (!events->empty())
    events->pop();

  ++descriptor.generation;
  ASSERT_EQ(make_job_request(*request_message, 411u, descriptor, 0u, input),
            ProtocolResult::kSuccess);
  ASSERT_EQ(pool.submit(*request_message), JobSubmitResult::kAccepted);
  ASSERT_NE(events->front(), nullptr);
  EXPECT_EQ(events->front()->envelope.sequence, 411u);
  EXPECT_EQ(events->front()->metadata.event, JobEventKind::kAdmitted);
  events->pop();
  ASSERT_NE(events->front(), nullptr);
  EXPECT_EQ(events->front()->envelope.sequence, 411u);
  EXPECT_EQ(events->front()->metadata.event, JobEventKind::kDispatched);

  completed = nullptr;
  const auto replacement_timeout = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (completed == nullptr && std::chrono::steady_clock::now() < replacement_timeout) {
    completed = pool.ready_result();
    if (completed == nullptr)
      std::this_thread::yield();
  }
  ASSERT_NE(completed, nullptr);
  pool.release_ready_result();
  EXPECT_TRUE(pool.release_session(descriptor.session_id));
}

TEST(JobWorkerPool, ReservesCapacityAndOrdersEqualDeadlinesByServiceClass)
{
  JobDescriptor descriptor = test_job_descriptor(30u);
  descriptor.kind = JobKind::kStreaming;
  descriptor.deadline_ns = 0u;
  descriptor.total_work_units = DrainBackend::kTotalSteps;
  std::atomic_bool boundary_reached{false};
  std::atomic_bool release_boundary{false};
  DrainBackend backend{.boundary_reached = &boundary_reached,
                       .release_boundary = &release_boundary,
                       .gate_first_boundary = true};
  std::array<JobAdapterRegistration, 1> entries{};
  std::array<JobAdapterRegistry, 1> registries{JobAdapterRegistry{entries}};
  ASSERT_TRUE(registries[0].add(make_routed_adapter(backend, job_route(descriptor),
                                                    sizeof(std::uint64_t), sizeof(std::uint64_t))));
  registries[0].freeze();
  auto created =
      JobWorkerPool::create(registries, 3uz, {}, 8uz, 1uz, WorkerPlacement::kNone, nullptr, 0uz,
                            JobQosPolicy{.interactive_reserve_slots = 1uz,
                                         .critical_reserve_slots = 1uz});
  ASSERT_TRUE(created) << created.error();
  JobWorkerPool pool = std::move(*created);

  std::array<std::byte, sizeof(std::uint64_t)> input{};
  StateWriter writer{input};
  ASSERT_TRUE(writer.write(std::uint64_t{3u}));
  const auto make_request = [&](std::uint64_t sequence, std::uint64_t session,
                                JobServiceClass service_class) {
    auto message = std::make_unique<JobRequestMessage>();
    JobDescriptor request_descriptor = descriptor;
    request_descriptor.session_id = session;
    EXPECT_EQ(make_job_request(*message, sequence, request_descriptor, 0u, input, service_class),
              ProtocolResult::kSuccess);
    return message;
  };

  auto blocker = make_request(500u, 50u, JobServiceClass::kBestEffort);
  ASSERT_EQ(pool.submit(*blocker), JobSubmitResult::kAccepted);
  const auto boundary_timeout = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (!boundary_reached.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < boundary_timeout)
    std::this_thread::yield();
  ASSERT_TRUE(boundary_reached.load(std::memory_order_acquire));

  auto best_effort = make_request(501u, 51u, JobServiceClass::kBestEffort);
  auto displaced = make_request(502u, 52u, JobServiceClass::kBestEffort);
  auto interactive = make_request(503u, 53u, JobServiceClass::kInteractive);
  auto critical = make_request(504u, 54u, JobServiceClass::kCritical);
  JobResultMessage rejection{};
  EXPECT_EQ(pool.submit(*best_effort), JobSubmitResult::kAccepted);
  EXPECT_EQ(pool.submit(*displaced, 0u, &rejection), JobSubmitResult::kQosCapacity);
  EXPECT_EQ(job_result_code(rejection), JobResultCode::kRejectedQos);
  EXPECT_EQ(rejection.metadata.service_class, JobServiceClass::kBestEffort);
  EXPECT_EQ(pool.submit(*interactive), JobSubmitResult::kAccepted);
  EXPECT_EQ(pool.submit(*critical), JobSubmitResult::kAccepted);
  EXPECT_EQ(pool.qos_rejection_count(), 1u);

  release_boundary.store(true, std::memory_order_release);
  release_boundary.notify_one();
  constexpr std::array expected_order{500u, 504u, 503u, 501u};
  for (const std::uint64_t sequence : expected_order) {
    const JobResultMessage* result{};
    const auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds{2};
    while (result == nullptr && std::chrono::steady_clock::now() < timeout) {
      result = pool.ready_result();
      if (result == nullptr)
        std::this_thread::yield();
    }
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->envelope.sequence, sequence);
    EXPECT_EQ(job_result_code(*result), JobResultCode::kComplete);
    pool.release_ready_result();
  }
}

TEST(JobWorkerPool, QuarantinesRepeatedlyFailingLaneUntilExplicitRecovery)
{
  JobDescriptor descriptor = test_job_descriptor(40u);
  descriptor.kind = JobKind::kIterative;
  descriptor.deadline_ns = 0u;
  descriptor.total_work_units = 1u;
  FailingRoutedBackend backend{};
  std::array<JobAdapterRegistration, 1> entries{};
  std::array<JobAdapterRegistry, 1> registries{JobAdapterRegistry{entries}};
  ASSERT_TRUE(registries[0].add(make_routed_adapter(backend, job_route(descriptor), 0uz, 0uz)));
  registries[0].freeze();
  auto created =
      JobWorkerPool::create(registries, 2uz, {}, 4uz, 1uz, WorkerPlacement::kNone, nullptr, 0uz, {},
                            WorkerSupervisionPolicy{.consecutive_failure_threshold = 2uz});
  ASSERT_TRUE(created) << created.error();
  JobWorkerPool pool = std::move(*created);

  const auto fail_once = [&](std::uint64_t sequence, std::uint64_t session) {
    JobDescriptor request_descriptor = descriptor;
    request_descriptor.session_id = session;
    JobRequestMessage request_message{};
    EXPECT_EQ(make_job_request(request_message, sequence, request_descriptor, 0u, {},
                               JobServiceClass::kInteractive),
              ProtocolResult::kSuccess);
    EXPECT_EQ(pool.submit(request_message), JobSubmitResult::kAccepted);
    const JobResultMessage* result{};
    const auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds{2};
    while (result == nullptr && std::chrono::steady_clock::now() < timeout) {
      result = pool.ready_result();
      if (result == nullptr)
        std::this_thread::yield();
    }
    EXPECT_NE(result, nullptr);
    if (result != nullptr) {
      EXPECT_EQ(job_result_code(*result), JobResultCode::kFailed);
      EXPECT_EQ(result->metadata.service_class, JobServiceClass::kInteractive);
      pool.release_ready_result();
    }
  };

  fail_once(600u, 60u);
  EXPECT_EQ(pool.failure_count(), 1u);
  EXPECT_EQ(pool.worker_consecutive_failures(0uz), 1uz);
  EXPECT_FALSE(pool.worker_quarantined(0uz));
  fail_once(601u, 61u);
  EXPECT_EQ(pool.failure_count(), 2u);
  EXPECT_EQ(pool.quarantine_count(), 1u);
  EXPECT_TRUE(pool.worker_quarantined(0uz));
  EXPECT_TRUE(pool.worker_drained(0uz));
  EXPECT_EQ(pool.recover_worker(1uz), WorkerRecoveryResult::kInvalidWorker);

  JobTransportStats transport_stats{};
  const std::string request_name = unique_name("fe-sv-ctrl");
  const std::string response_name = unique_name("fe-sv-stat");
  auto service_result =
      JobControlService::create(request_name, response_name, pool, transport_stats, 2u);
  ASSERT_TRUE(service_result) << service_result.error();
  JobControlService service = std::move(*service_result);
  auto client_result = JobControlClient::connect(request_name, response_name);
  ASSERT_TRUE(client_result) << client_result.error();
  JobControlClient client = std::move(*client_result);
  JobControlResponse response{};
  const auto administer = [&](JobControlOperation operation) {
    const JobControlRequest request = make_job_control_request(700u, 70u, operation, 0u);
    EXPECT_EQ(client.try_submit(request), ClientResult::kSuccess);
    EXPECT_EQ(service.poll(), JobControlServiceResult::kProgress);
    EXPECT_EQ(client.try_receive(response), ClientResult::kSuccess);
  };
  const JobControlRequest status =
      make_job_control_request(699u, 70u, JobControlOperation::kStatus);
  ASSERT_EQ(client.try_submit(status), ClientResult::kSuccess);
  ASSERT_EQ(service.poll(), JobControlServiceResult::kProgress);
  ASSERT_EQ(client.try_receive(response), ClientResult::kSuccess);
  EXPECT_EQ(response.metadata.status.quarantined_mask, 1u);
  EXPECT_EQ(response.metadata.status.worker_quarantines, 1u);
  administer(JobControlOperation::kResumeWorker);
  EXPECT_EQ(response.metadata.code, JobControlCode::kWorkerQuarantined);
  administer(JobControlOperation::kRecoverWorker);
  EXPECT_EQ(response.metadata.code, JobControlCode::kSuccess);
  EXPECT_FALSE(pool.worker_quarantined(0uz));
  EXPECT_FALSE(pool.worker_draining(0uz));
  EXPECT_EQ(pool.accepting_worker_count(), 1uz);
  EXPECT_EQ(pool.recover_worker(0uz), WorkerRecoveryResult::kNotQuarantined);
}

TEST(JobWorkerPool, DrainsActiveLaneByMigratingAtWorkBoundary)
{
  JobDescriptor descriptor = test_job_descriptor(9u);
  descriptor.kind = JobKind::kStreaming;
  descriptor.deadline_ns = 0u;
  descriptor.total_work_units = DrainBackend::kTotalSteps;
  std::atomic_bool boundary_reached{false};
  std::atomic_bool release_boundary{false};
  DrainBackend source{.boundary_reached = &boundary_reached,
                      .release_boundary = &release_boundary,
                      .gate_first_boundary = true};
  DrainBackend destination{};
  std::array<JobAdapterRegistration, 1> source_entries{};
  std::array<JobAdapterRegistration, 1> destination_entries{};
  std::array<JobAdapterRegistry, 2> registries{JobAdapterRegistry{source_entries},
                                               JobAdapterRegistry{destination_entries}};
  ASSERT_TRUE(registries[0].add(make_routed_adapter(source, job_route(descriptor),
                                                    sizeof(std::uint64_t), sizeof(std::uint64_t))));
  ASSERT_TRUE(registries[1].add(make_routed_adapter(destination, job_route(descriptor),
                                                    sizeof(std::uint64_t), sizeof(std::uint64_t))));
  for (JobAdapterRegistry& registry : registries)
    registry.freeze();
  auto events = JobEventBuffer::create(16uz);
  ASSERT_TRUE(events) << events.error();
  auto created =
      JobWorkerPool::create(registries, 2uz, {}, 2uz, 1uz, WorkerPlacement::kNone, &*events, 256uz);
  ASSERT_TRUE(created) << created.error();
  JobWorkerPool pool = std::move(*created);
  EXPECT_EQ(pool.migration_capacity_bytes(), 256uz);
  EXPECT_EQ(pool.accepting_worker_count(), 2uz);

  std::array<std::byte, sizeof(std::uint64_t)> input{};
  StateWriter writer{input};
  ASSERT_TRUE(writer.write(std::uint64_t{3u}));
  JobRequestMessage request_message{};
  ASSERT_EQ(make_job_request(request_message, 601u, descriptor, 0u, input),
            ProtocolResult::kSuccess);
  ASSERT_EQ(pool.submit(request_message), JobSubmitResult::kAccepted);
  const auto boundary_timeout = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (!boundary_reached.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < boundary_timeout)
    std::this_thread::yield();
  ASSERT_TRUE(boundary_reached.load(std::memory_order_acquire));
  EXPECT_EQ(pool.request_worker_drain(0uz), WorkerDrainResult::kStarted);
  EXPECT_EQ(pool.request_worker_drain(0uz), WorkerDrainResult::kAlreadyDraining);
  EXPECT_EQ(pool.request_worker_drain(9uz), WorkerDrainResult::kInvalidWorker);
  release_boundary.store(true, std::memory_order_release);
  release_boundary.notify_one();

  const JobResultMessage* result{};
  const auto result_timeout = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (result == nullptr && std::chrono::steady_clock::now() < result_timeout) {
    result = pool.ready_result();
    if (result == nullptr)
      std::this_thread::yield();
  }
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(validate(*result), ProtocolResult::kSuccess);
  EXPECT_EQ(job_result_code(*result), JobResultCode::kComplete);
  EXPECT_EQ(result->metadata.progress.completed_work_units, DrainBackend::kTotalSteps);
  StateReader result_reader{result->payload_values()};
  const auto migrated_value = result_reader.read<std::uint64_t>();
  ASSERT_TRUE(migrated_value);
  EXPECT_EQ(*migrated_value, destination.value);
  EXPECT_EQ(source.completed, 1u);
  EXPECT_EQ(destination.completed, DrainBackend::kTotalSteps);
  EXPECT_TRUE(pool.worker_drained(0uz));
  EXPECT_TRUE(pool.worker_draining(0uz));
  EXPECT_EQ(pool.accepting_worker_count(), 1uz);
  pool.release_ready_result();
  EXPECT_TRUE(pool.release_session(descriptor.session_id));

  JobMetrics metrics{};
  while (const JobEventMessage* event = events->front()) {
    EXPECT_EQ(validate(*event), ProtocolResult::kSuccess);
    metrics.record(*event);
    events->pop();
  }
  EXPECT_EQ(metrics.counters().migrations_started, 1u);
  EXPECT_EQ(metrics.counters().migrations_completed, 1u);
  EXPECT_EQ(metrics.counters().completed, 1u);
  EXPECT_EQ(pool.request_worker_drain(1uz), WorkerDrainResult::kStarted);
  EXPECT_TRUE(pool.worker_drained(1uz));
  JobResultMessage rejection{};
  EXPECT_EQ(pool.submit(request_message, 0u, &rejection), JobSubmitResult::kFull);
  EXPECT_EQ(job_result_code(rejection), JobResultCode::kRejectedCapacity);
  EXPECT_TRUE(pool.resume_worker(0uz));
  EXPECT_TRUE(pool.resume_worker(1uz));
  EXPECT_FALSE(pool.worker_draining(0uz));
  EXPECT_EQ(pool.accepting_worker_count(), 2uz);
  EXPECT_FALSE(pool.resume_worker(0uz));
}

TEST(JobWorkerPool, RefusesToDrainTheOnlyLaneForAcceptedWork)
{
  JobDescriptor descriptor = test_job_descriptor(10u);
  descriptor.kind = JobKind::kStreaming;
  descriptor.deadline_ns = 0u;
  descriptor.total_work_units = DrainBackend::kTotalSteps;
  std::atomic_bool boundary_reached{false};
  std::atomic_bool release_boundary{false};
  DrainBackend backend{.boundary_reached = &boundary_reached,
                       .release_boundary = &release_boundary,
                       .gate_first_boundary = true};
  std::array<JobAdapterRegistration, 1> entries{};
  std::array<JobAdapterRegistry, 1> registries{JobAdapterRegistry{entries}};
  ASSERT_TRUE(registries[0].add(make_routed_adapter(backend, job_route(descriptor),
                                                    sizeof(std::uint64_t), sizeof(std::uint64_t))));
  registries[0].freeze();
  auto created =
      JobWorkerPool::create(registries, 2uz, {}, 2uz, 1uz, WorkerPlacement::kNone, nullptr, 256uz);
  ASSERT_TRUE(created) << created.error();
  JobWorkerPool pool = std::move(*created);

  std::array<std::byte, sizeof(std::uint64_t)> input{};
  StateWriter writer{input};
  ASSERT_TRUE(writer.write(std::uint64_t{4u}));
  JobRequestMessage first{};
  JobRequestMessage second{};
  ASSERT_EQ(make_job_request(first, 701u, descriptor, 0u, input), ProtocolResult::kSuccess);
  ++descriptor.session_id;
  ASSERT_EQ(make_job_request(second, 702u, descriptor, 0u, input), ProtocolResult::kSuccess);
  ASSERT_EQ(pool.submit(first), JobSubmitResult::kAccepted);
  ASSERT_EQ(pool.submit(second), JobSubmitResult::kAccepted);
  const auto boundary_timeout = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (!boundary_reached.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < boundary_timeout)
    std::this_thread::yield();
  ASSERT_TRUE(boundary_reached.load(std::memory_order_acquire));
  EXPECT_EQ(pool.request_worker_drain(0uz), WorkerDrainResult::kWouldStrandWork);
  EXPECT_FALSE(pool.worker_draining(0uz));
  release_boundary.store(true, std::memory_order_release);
  release_boundary.notify_one();

  std::size_t completed{};
  const auto result_timeout = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (completed < 2uz && std::chrono::steady_clock::now() < result_timeout) {
    const JobResultMessage* const result = pool.ready_result();
    if (result == nullptr) {
      std::this_thread::yield();
      continue;
    }
    EXPECT_EQ(job_result_code(*result), JobResultCode::kComplete);
    pool.release_ready_result();
    ++completed;
  }
  EXPECT_EQ(completed, 2uz);
  EXPECT_TRUE(pool.release_session(first.metadata.descriptor.session_id));
  EXPECT_TRUE(pool.release_session(second.metadata.descriptor.session_id));
}

TEST(JobWorkerPool, FinishesActiveWorkLocallyWhenLiveHandoffIsDisabled)
{
  JobDescriptor descriptor = test_job_descriptor(11u);
  descriptor.kind = JobKind::kStreaming;
  descriptor.deadline_ns = 0u;
  descriptor.total_work_units = DrainBackend::kTotalSteps;
  std::atomic_bool boundary_reached{false};
  std::atomic_bool release_boundary{false};
  DrainBackend backend{.boundary_reached = &boundary_reached,
                       .release_boundary = &release_boundary,
                       .gate_first_boundary = true};
  std::array<JobAdapterRegistration, 1> entries{};
  std::array<JobAdapterRegistry, 1> registries{JobAdapterRegistry{entries}};
  ASSERT_TRUE(registries[0].add(make_routed_adapter(backend, job_route(descriptor),
                                                    sizeof(std::uint64_t), sizeof(std::uint64_t))));
  registries[0].freeze();
  auto created = JobWorkerPool::create(registries, 1uz, {}, 1uz);
  ASSERT_TRUE(created) << created.error();
  JobWorkerPool pool = std::move(*created);
  EXPECT_EQ(pool.migration_capacity_bytes(), 0uz);

  std::array<std::byte, sizeof(std::uint64_t)> input{};
  StateWriter writer{input};
  ASSERT_TRUE(writer.write(std::uint64_t{4u}));
  JobRequestMessage request_message{};
  ASSERT_EQ(make_job_request(request_message, 703u, descriptor, 0u, input),
            ProtocolResult::kSuccess);
  ASSERT_EQ(pool.submit(request_message), JobSubmitResult::kAccepted);
  const auto boundary_timeout = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (!boundary_reached.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < boundary_timeout)
    std::this_thread::yield();
  ASSERT_TRUE(boundary_reached.load(std::memory_order_acquire));
  EXPECT_EQ(pool.request_worker_drain(0uz), WorkerDrainResult::kStarted);
  release_boundary.store(true, std::memory_order_release);
  release_boundary.notify_one();

  const JobResultMessage* result{};
  const auto result_timeout = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (result == nullptr && std::chrono::steady_clock::now() < result_timeout) {
    result = pool.ready_result();
    if (result == nullptr)
      std::this_thread::yield();
  }
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(job_result_code(*result), JobResultCode::kComplete);
  EXPECT_EQ(backend.completed, DrainBackend::kTotalSteps);
  EXPECT_FALSE(pool.worker_drained(0uz));
  pool.release_ready_result();
  EXPECT_TRUE(pool.worker_drained(0uz));
  EXPECT_TRUE(pool.release_session(descriptor.session_id));
  EXPECT_TRUE(pool.resume_worker(0uz));
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

  const std::string request_name = unique_name("fe-svc-req");
  const std::string result_name = unique_name("fe-svc-res");
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
  ASSERT_EQ(make_job_request(*request_message, 500u, descriptor, 0u, input,
                             JobServiceClass::kCritical),
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
  EXPECT_EQ(result->metadata.service_class, JobServiceClass::kCritical);

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

TEST(JobControl, ReportsStatusAndAdministersWorkerLifecycle)
{
  JobDescriptor descriptor = test_job_descriptor(15u);
  descriptor.kind = JobKind::kIterative;
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

  JobTransportStats transport_stats{.requests_received = 7u,
                                    .requests_accepted = 5u,
                                    .requests_rejected = 2u,
                                    .results_published = 6u};
  const std::string request_name = unique_name("fe-ctrl-req");
  const std::string response_name = unique_name("fe-ctrl-res");
  auto created_service =
      JobControlService::create(request_name, response_name, pool, transport_stats, 2u);
  ASSERT_TRUE(created_service) << created_service.error();
  JobControlService service = std::move(*created_service);
  auto connected = JobControlClient::connect(request_name, response_name);
  ASSERT_TRUE(connected) << connected.error();
  JobControlClient client = std::move(*connected);

  JobControlRequest request = make_job_control_request(1u, 99u, JobControlOperation::kStatus);
  EXPECT_EQ(validate(request), ProtocolResult::kSuccess);
  ASSERT_EQ(client.try_submit(request), ClientResult::kSuccess);
  ASSERT_EQ(service.poll(), JobControlServiceResult::kProgress);
  JobControlResponse response{};
  ASSERT_EQ(client.try_receive(response), ClientResult::kSuccess);
  EXPECT_EQ(response.metadata.code, JobControlCode::kSuccess);
  EXPECT_EQ(response.metadata.status.worker_count, 1u);
  EXPECT_EQ(response.metadata.status.accepting_workers, 1u);
  EXPECT_EQ(response.metadata.status.requests_accepted, 5u);
  JobControlResponse invalid_response = response;
  invalid_response.metadata.status.worker_count = kMaxJobControlWorkers + 1u;
  EXPECT_EQ(validate(invalid_response), ProtocolResult::kInvalidMetadata);
  invalid_response = response;
  invalid_response.metadata.status.drained_mask = 1u;
  EXPECT_EQ(validate(invalid_response), ProtocolResult::kInvalidMetadata);
  invalid_response = response;
  invalid_response.metadata.status.quarantined_mask = 1u;
  EXPECT_EQ(validate(invalid_response), ProtocolResult::kInvalidMetadata);

  request = make_job_control_request(20u, 99u, JobControlOperation::kRecoverWorker, 0u);
  ASSERT_EQ(client.try_submit(request), ClientResult::kSuccess);
  ASSERT_EQ(service.poll(), JobControlServiceResult::kProgress);
  ASSERT_EQ(client.try_receive(response), ClientResult::kSuccess);
  EXPECT_EQ(response.metadata.code, JobControlCode::kWorkerNotQuarantined);

  request = make_job_control_request(2u, 99u, JobControlOperation::kResumeWorker, 0u);
  ASSERT_EQ(client.try_submit(request), ClientResult::kSuccess);
  ASSERT_EQ(service.poll(), JobControlServiceResult::kProgress);
  ASSERT_EQ(client.try_receive(response), ClientResult::kSuccess);
  EXPECT_EQ(response.metadata.code, JobControlCode::kWorkerNotDrained);

  request = make_job_control_request(3u, 99u, JobControlOperation::kDrainWorker, 0u);
  ASSERT_EQ(client.try_submit(request), ClientResult::kSuccess);
  ASSERT_EQ(service.poll(), JobControlServiceResult::kProgress);
  ASSERT_EQ(client.try_receive(response), ClientResult::kSuccess);
  EXPECT_EQ(response.metadata.code, JobControlCode::kSuccess);
  EXPECT_EQ(response.metadata.status.accepting_workers, 0u);
  EXPECT_EQ(response.metadata.status.draining_mask, 1u);
  EXPECT_EQ(response.metadata.status.drained_mask, 1u);

  request = make_job_control_request(4u, 99u, JobControlOperation::kResumeWorker, 0u);
  ASSERT_EQ(client.try_submit(request), ClientResult::kSuccess);
  ASSERT_EQ(service.poll(), JobControlServiceResult::kProgress);
  ASSERT_EQ(client.try_receive(response), ClientResult::kSuccess);
  EXPECT_EQ(response.metadata.code, JobControlCode::kSuccess);
  EXPECT_EQ(response.metadata.status.accepting_workers, 1u);

  request = make_job_control_request(5u, 99u, JobControlOperation::kDrainWorker, 9u);
  ASSERT_EQ(client.try_submit(request), ClientResult::kSuccess);
  ASSERT_EQ(service.poll(), JobControlServiceResult::kProgress);
  ASSERT_EQ(client.try_receive(response), ClientResult::kSuccess);
  EXPECT_EQ(response.metadata.code, JobControlCode::kInvalidWorker);

  JobControlRequest invalid = make_job_control_request(6u, 99u, JobControlOperation::kStatus, 0u);
  EXPECT_EQ(validate(invalid), ProtocolResult::kInvalidMetadata);
  EXPECT_EQ(client.try_submit(invalid), ClientResult::kInvalidRequest);

  request = make_job_control_request(7u, 99u, JobControlOperation::kShutdown);
  ASSERT_EQ(client.try_submit(request), ClientResult::kSuccess);
  ASSERT_EQ(service.poll(), JobControlServiceResult::kStopped);
  ASSERT_EQ(client.try_receive(response), ClientResult::kSuccess);
  EXPECT_EQ(response.metadata.operation, JobControlOperation::kShutdown);
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

  admitted.metadata.service_class = static_cast<JobServiceClass>(99u);
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
  record(JobEventDetails{.event = JobEventKind::kRejected,
                         .descriptor = descriptor,
                         .progress = JobProgress{.state = JobState::kFailed,
                                                 .completed_work_units = 0u,
                                                 .remaining_work_units = 4u},
                         .timestamp_ns = 1'110u,
                         .result_code = JobResultCode::kRejectedQos,
                         .service_class = JobServiceClass::kBestEffort});
  metrics.observe_workers(2uz, 1u, 1u);
  metrics.record_event_drops(2u);

  EXPECT_EQ(metrics.counters().events, 6u);
  EXPECT_EQ(metrics.counters().preempted, 1u);
  EXPECT_EQ(metrics.counters().migrations_completed, 1u);
  EXPECT_EQ(metrics.counters().completed_work_units, 4u);
  EXPECT_EQ(metrics.counters().queue_high_watermark, 3u);
  EXPECT_EQ(metrics.counters().busy_workers_high_watermark, 2u);
  EXPECT_EQ(metrics.counters().events_dropped, 2u);
  EXPECT_EQ(metrics.counters().rejected_qos, 1u);
  EXPECT_EQ(metrics.counters().worker_quarantines, 1u);
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
  EXPECT_NE(prometheus.str().find("flowedge_relay_job_rejected_qos_total 1"), std::string::npos);
  std::ostringstream json{};
  write_metrics_json(json, metrics);
  EXPECT_NE(json.str().find("\"streaming\":{\"completed\":1"), std::string::npos);
  EXPECT_NE(json.str().find("\"worker_quarantines\":1"), std::string::npos);
  std::ostringstream otlp{};
  write_otlp_json(otlp, metrics);
  EXPECT_NE(otlp.str().find("\"job.kind\""), std::string::npos);
  EXPECT_NE(otlp.str().find("flowedge.relay.job.execution_latency"), std::string::npos);
}

} // namespace
