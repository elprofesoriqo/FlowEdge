#include "api/engine.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <thread>
#include <vector>

namespace {

TEST(CAbiContract, StatusValuesAreStable)
{
  EXPECT_EQ(FE_STATUS_OK, 0);
  EXPECT_EQ(FE_STATUS_INVALID_ARGUMENT, 1);
  EXPECT_EQ(FE_STATUS_RESOURCE_EXHAUSTED, 2);
  EXPECT_EQ(FE_STATUS_TOKEN_OUT_OF_RANGE, 3);
  EXPECT_EQ(FE_STATUS_UNSUPPORTED_MODEL, 4);
  EXPECT_EQ(FE_STATUS_INVALID_SOLVER, 5);
  EXPECT_EQ(FE_STATUS_NO_ACTIVE_OPERATION, 6);
  EXPECT_EQ(FE_STATUS_STATE_UNAVAILABLE, 7);
  EXPECT_EQ(FE_STATUS_CANCELLED, 8);
  EXPECT_EQ(FE_STATUS_INCOMPATIBLE_STATE, 9);
  EXPECT_EQ(FE_STATUS_CORRUPT_STATE, 10);
}

TEST(CAbiContract, NullHandlesReportInvalidArguments)
{
  std::int32_t token{};
  float value{};
  std::size_t remaining{};

  EXPECT_EQ(fe_engine_run(nullptr, &token, 1uz, &value), FE_STATUS_INVALID_ARGUMENT);
  EXPECT_STREQ(fe_engine_last_error(), "Invalid arguments to fe_engine_run");
  EXPECT_EQ(fe_engine_step(nullptr, token, &value), FE_STATUS_INVALID_ARGUMENT);
  EXPECT_STREQ(fe_engine_last_error(), "Invalid null arguments to fe_engine_step");
  EXPECT_EQ(fe_engine_flow_advance(nullptr, 1uz, &value, &remaining), FE_STATUS_INVALID_ARGUMENT);
  EXPECT_STREQ(fe_engine_last_error(), "Invalid arguments to fe_engine_flow_advance");
  EXPECT_EQ(fe_engine_export_decode_state(nullptr, &value, sizeof(value)),
            FE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(fe_engine_import_decode_state(nullptr, &value, sizeof(value)),
            FE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(fe_engine_decode_state_bytes(nullptr), 0uz);
  EXPECT_EQ(fe_engine_action_dim(nullptr), 0uz);
  EXPECT_EQ(fe_engine_condition_dim(nullptr), 0uz);
  EXPECT_EQ(fe_weights_size_bytes(nullptr), 0uz);
}

TEST(CAbiContract, LastErrorIsThreadLocal)
{
  ASSERT_EQ(fe_engine_load(nullptr), nullptr);
  const std::string main_error{fe_engine_last_error()};
  std::string worker_error;
  std::thread worker{[&] {
    static_cast<void>(fe_engine_step(nullptr, 0, nullptr));
    worker_error = fe_engine_last_error();
  }};
  worker.join();

  EXPECT_EQ(main_error, "Invalid null model path");
  EXPECT_EQ(worker_error, "Invalid null arguments to fe_engine_step");
  EXPECT_STREQ(fe_engine_last_error(), main_error.c_str());
}

#ifdef FLOWEDGE_SOURCE_DIR
TEST(CAbiContract, DecodeBuffersRejectUndersizedStorage)
{
  const std::filesystem::path model =
      std::filesystem::path{FLOWEDGE_SOURCE_DIR} / "models" / "mamba_flow.safetensors";
  if (!std::filesystem::exists(model))
    GTEST_SKIP() << "models/mamba_flow.safetensors is not available";

  fe_engine* engine = fe_engine_load_with_threads(model.string().c_str(), 0u);
  ASSERT_NE(engine, nullptr) << fe_engine_last_error();
  const std::size_t bytes = fe_engine_decode_state_bytes(engine);
  ASSERT_GT(bytes, 1uz);
  std::vector<std::byte> undersized(bytes - 1uz);
  EXPECT_EQ(fe_engine_export_decode_state(engine, undersized.data(), undersized.size()),
            FE_STATUS_INVALID_ARGUMENT);
  EXPECT_NE(std::string{fe_engine_last_error()}.find("too small"), std::string::npos);
  EXPECT_EQ(fe_engine_import_decode_state(engine, undersized.data(), undersized.size()),
            FE_STATUS_INVALID_ARGUMENT);
  EXPECT_NE(std::string{fe_engine_last_error()}.find("truncated"), std::string::npos);
  fe_engine_free(engine);
}
#endif

} // namespace
