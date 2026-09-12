#include "relay_test_utils.h"

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

TEST(ActionChunks, BuildsVersionedZeroCopyViewAndRejectsInvalidInput)
{
  const fe_model_metadata model = test_model(4uz, 8uz);
  constexpr std::array values{0.0F, 0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F, 0.7F};
  ActionMessage action = complete_action(model, values, 8u, 9u, 3u, 100u, 300u);
  const auto chunk = make_action_chunk(action, 2uz, 150u, 160u, 10u);
  ASSERT_TRUE(chunk);
  EXPECT_EQ(chunk->descriptor.protocol_version, kActionChunkProtocolVersion);
  EXPECT_EQ(chunk->descriptor.control_dim, 2u);
  EXPECT_EQ(chunk->descriptor.step_count, 4u);
  EXPECT_EQ(chunk->descriptor.valid_until_ns, 200u);
  EXPECT_EQ(chunk->descriptor.sequence, 8u);
  EXPECT_EQ(chunk->descriptor.session_id, 9u);
  ASSERT_EQ(chunk->step(1uz).size(), 2uz);
  EXPECT_FLOAT_EQ(chunk->step(1uz)[0], 0.2F);
  EXPECT_TRUE(chunk->step(4uz).empty());

  EXPECT_EQ(make_action_chunk(action, 3uz, 150u, 160u, 10u).error(),
            ActionChunkError::kInvalidShape);
  EXPECT_EQ(make_action_chunk(action, 2uz, 301u, 301u, 10u).error(), ActionChunkError::kExpired);
  action.metadata.status = FE_ACTION_RUNNING;
  EXPECT_EQ(make_action_chunk(action, 2uz, 150u, 160u, 10u).error(),
            ActionChunkError::kIncompleteAction);
  action.metadata.status = FE_ACTION_COMPLETE;
  action.action[3] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(make_action_chunk(action, 2uz, 150u, 160u, 10u).error(), ActionChunkError::kNonFinite);
}

TEST(ActionDelivery, ReplacesAndBlendsFreshChunksByControlTime)
{
  const fe_model_metadata model = test_model(4uz, 8uz);
  constexpr std::array lower{-1.0F, -1.0F};
  constexpr std::array upper{1.0F, 1.0F};
  constexpr std::array delta{1.0F, 1.0F};
  auto created = ActionDeliveryGate::create(model,
                                            ActionDeliveryConfig{.control_dim = 2uz,
                                                                 .overlap_steps = 2uz,
                                                                 .step_period_ns = 10u,
                                                                 .max_source_age_ns = 100u},
                                            ActionSafetyLimits{.lower = lower,
                                                               .upper = upper,
                                                               .max_delta_per_step = delta});
  ASSERT_TRUE(created) << to_string(created.error());
  ActionDeliveryGate gate = std::move(*created);

  constexpr std::array old_values{0.0F, 0.0F, 0.3F, 0.3F, 0.6F, 0.6F, 0.9F, 0.9F};
  const ActionMessage old = complete_action(model, old_values, 1u, 9u, 1u, 100u);
  EXPECT_EQ(gate.accept(old, 100u, 100u), ActionChunkAcceptResult::kAccepted);
  const auto first = gate.next(100u);
  ASSERT_TRUE(first);
  EXPECT_EQ(first->step_index, 0u);
  EXPECT_FLOAT_EQ(first->values[0], 0.0F);
  EXPECT_FALSE(first->blended);
  EXPECT_EQ(gate.next(100u).error(), ActionStepError::kNotReady);

  constexpr std::array new_values{-0.3F, -0.3F, -0.6F, -0.6F, -0.9F, -0.9F, -1.0F, -1.0F};
  const ActionMessage replacement = complete_action(model, new_values, 2u, 9u, 2u, 110u);
  EXPECT_EQ(gate.accept(replacement, 110u, 110u), ActionChunkAcceptResult::kReplaced);
  const auto blended_first = gate.next(110u);
  ASSERT_TRUE(blended_first);
  EXPECT_TRUE(blended_first->blended);
  EXPECT_NEAR(blended_first->values[0], 0.1F, 1.0e-6F);
  const auto blended_second = gate.next(120u);
  ASSERT_TRUE(blended_second);
  EXPECT_TRUE(blended_second->blended);
  EXPECT_NEAR(blended_second->values[0], -0.2F, 1.0e-6F);

  EXPECT_EQ(gate.accept(old, 121u, 121u), ActionChunkAcceptResult::kStale);
  const ActionMessage other_session = complete_action(model, new_values, 3u, 10u, 3u, 121u);
  EXPECT_EQ(gate.accept(other_session, 121u, 121u), ActionChunkAcceptResult::kSessionMismatch);
  const auto skipped = gate.next(130u);
  ASSERT_TRUE(skipped);
  EXPECT_EQ(skipped->step_index, 2u);
  EXPECT_EQ(gate.counters().accepted, 2u);
  EXPECT_EQ(gate.counters().replaced, 1u);
  EXPECT_EQ(gate.counters().blended, 2u);
  EXPECT_EQ(gate.counters().rejected_stale, 1u);
  EXPECT_EQ(gate.counters().rejected_session, 1u);
}

TEST(ActionDelivery, RejectsOrClampsUnsafeAndExpiredActions)
{
  const fe_model_metadata model = test_model(4uz, 4uz);
  constexpr std::array lower{-1.0F, -1.0F};
  constexpr std::array upper{1.0F, 1.0F};
  constexpr std::array delta{0.2F, 0.2F};
  constexpr std::array unsafe_values{2.0F, 0.0F, -1.0F, 0.0F};
  const ActionMessage unsafe = complete_action(model, unsafe_values, 1u, 9u, 1u, 100u);
  auto reject_created =
      ActionDeliveryGate::create(model,
                                 ActionDeliveryConfig{.control_dim = 2uz,
                                                      .step_period_ns = 10u,
                                                      .max_source_age_ns = 20u,
                                                      .unsafe_policy = UnsafeActionPolicy::kReject},
                                 ActionSafetyLimits{.lower = lower,
                                                    .upper = upper,
                                                    .max_delta_per_step = delta});
  ASSERT_TRUE(reject_created);
  EXPECT_EQ(reject_created->accept(unsafe, 100u, 100u), ActionChunkAcceptResult::kUnsafe);

  constexpr std::array safe_values{0.0F, 0.0F, 0.8F, 0.8F};
  const ActionMessage safe = complete_action(model, safe_values, 2u, 9u, 2u, 100u);
  EXPECT_EQ(reject_created->accept(safe, 100u, 100u), ActionChunkAcceptResult::kAccepted);
  ASSERT_TRUE(reject_created->next(100u));
  EXPECT_EQ(reject_created->next(110u).error(), ActionStepError::kUnsafe);

  auto clamp_created =
      ActionDeliveryGate::create(model,
                                 ActionDeliveryConfig{.control_dim = 2uz,
                                                      .step_period_ns = 10u,
                                                      .unsafe_policy = UnsafeActionPolicy::kClamp},
                                 ActionSafetyLimits{.lower = lower,
                                                    .upper = upper,
                                                    .max_delta_per_step = delta});
  ASSERT_TRUE(clamp_created);
  EXPECT_EQ(clamp_created->accept(unsafe, 100u, 100u), ActionChunkAcceptResult::kAccepted);
  const auto clamped_first = clamp_created->next(100u);
  ASSERT_TRUE(clamped_first);
  EXPECT_TRUE(clamped_first->clamped);
  EXPECT_FLOAT_EQ(clamped_first->values[0], 1.0F);
  const auto clamped_second = clamp_created->next(110u);
  ASSERT_TRUE(clamped_second);
  EXPECT_TRUE(clamped_second->clamped);
  EXPECT_NEAR(clamped_second->values[0], 0.8F, 1.0e-6F);

  reject_created->reset();
  const ActionMessage aging = complete_action(model, safe_values, 3u, 9u, 3u, 100u);
  EXPECT_EQ(reject_created->accept(aging, 110u, 110u), ActionChunkAcceptResult::kAccepted);
  EXPECT_EQ(reject_created->next(121u).error(), ActionStepError::kExpired);
  EXPECT_EQ(reject_created->counters().rejected_expired, 1u);
}

TEST(ActionDelivery, ValidatesConfigurationAndPreservesLastSafeCommandOnReject)
{
  const fe_model_metadata model = test_model(4uz, 4uz);
  constexpr std::array lower{-1.0F, -1.0F};
  constexpr std::array upper{1.0F, 1.0F};
  constexpr std::array delta{0.2F, 0.2F};
  const auto invalid_config = ActionDeliveryGate::create(
      model, ActionDeliveryConfig{.control_dim = 3uz, .step_period_ns = 10u},
      ActionSafetyLimits{.lower = lower, .upper = upper, .max_delta_per_step = delta});
  ASSERT_FALSE(invalid_config);
  EXPECT_EQ(invalid_config.error(), ActionDeliveryCreateError::kInvalidConfig);
  const auto invalid_overlap = ActionDeliveryGate::create(
      model, ActionDeliveryConfig{.control_dim = 2uz, .overlap_steps = 3uz, .step_period_ns = 10u},
      ActionSafetyLimits{.lower = lower, .upper = upper, .max_delta_per_step = delta});
  ASSERT_FALSE(invalid_overlap);
  EXPECT_EQ(invalid_overlap.error(), ActionDeliveryCreateError::kInvalidOverlap);

  auto created =
      ActionDeliveryGate::create(model,
                                 ActionDeliveryConfig{.control_dim = 2uz,
                                                      .step_period_ns = 10u,
                                                      .unsafe_policy = UnsafeActionPolicy::kReject},
                                 ActionSafetyLimits{.lower = lower,
                                                    .upper = upper,
                                                    .max_delta_per_step = delta});
  ASSERT_TRUE(created);
  constexpr std::array values{0.0F, 0.0F, 0.1F, 0.8F};
  const ActionMessage action = complete_action(model, values, 1u, 9u, 1u, 100u);
  ASSERT_EQ(created->accept(action, 100u, 100u), ActionChunkAcceptResult::kAccepted);
  const auto first = created->next(100u);
  ASSERT_TRUE(first);
  EXPECT_EQ(created->next(110u).error(), ActionStepError::kUnsafe);

  constexpr std::array recovery_values{0.25F, 0.0F, 0.25F, 0.0F};
  const ActionMessage recovery = complete_action(model, recovery_values, 2u, 9u, 2u, 110u);
  ASSERT_EQ(created->accept(recovery, 110u, 110u), ActionChunkAcceptResult::kAccepted);
  EXPECT_EQ(created->next(110u).error(), ActionStepError::kUnsafe);
}

} // namespace
