#include "loader/safetensors.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>

namespace {

class MetadataFixture : public ::testing::Test
{
protected:
  void TearDown() override { std::filesystem::remove(path_); }

  void write_header(std::string metadata_value)
  {
    const std::string header =
        "{\"__metadata__\":{\"flowedge.deployment_profile\":\"" + metadata_value + "\"}}";
    const auto path =
        std::filesystem::temp_directory_path() / "flowedge-deployment-profile.safetensors";
    path_ = path;
    std::ofstream output{path_, std::ios::binary | std::ios::trunc};
    ASSERT_TRUE(output.good());
    const std::uint64_t size = header.size();
    output.write(reinterpret_cast<const char*>(&size), sizeof(size));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
  }

  std::filesystem::path path_{};
};

TEST_F(MetadataFixture, ReadsAndUnescapesDeploymentProfile)
{
  write_header(
      "{\\\"profile_version\\\":1,\\\"model_compatibility_version\\\":1,"
      "\\\"observation_schema_hash\\\":\\\"sha256:"
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\\\","
      "\\\"action_dim\\\":1,\\\"action_horizon\\\":4,\\\"action_units\\\":\\\"normalized\\\","
      "\\\"normalization_type\\\":\\\"none\\\",\\\"normalization_parameters\\\":{},"
      "\\\"solver_default\\\":\\\"euler\\\",\\\"solver_min_steps\\\":1,\\\"solver_max_steps\\\":"
      "4}");
  const auto encoded =
      fe::safetensors_metadata_value(path_.string(), fe::kDeploymentProfileMetadataKey);
  ASSERT_TRUE(encoded.has_value());
  ASSERT_TRUE(encoded->has_value());
  const auto profile = fe::parse_deployment_profile(**encoded);
  ASSERT_TRUE(profile.has_value()) << fe::deployment_profile_error_message(profile.error());
  EXPECT_EQ(profile->action_dim, 1uz);
  EXPECT_EQ(profile->action_horizon, 4uz);
}

TEST_F(MetadataFixture, RejectsInvalidProfileBeforeLoadingWeights)
{
  write_header(
      "{\\\"profile_version\\\":99,\\\"model_compatibility_version\\\":1,"
      "\\\"observation_schema_hash\\\":\\\"sha256:"
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\\\","
      "\\\"action_dim\\\":1,\\\"action_horizon\\\":4,\\\"action_units\\\":\\\"normalized\\\","
      "\\\"normalization_type\\\":\\\"none\\\",\\\"normalization_parameters\\\":{},"
      "\\\"solver_default\\\":\\\"euler\\\",\\\"solver_min_steps\\\":1,\\\"solver_max_steps\\\":"
      "4}");
  const auto weights = fe::ModelWeights::open(path_.string());
  ASSERT_FALSE(weights.has_value());
  EXPECT_STREQ(weights.error(), "deployment profile version is unsupported");
}

TEST_F(MetadataFixture, MissingProfileIsNotAnError)
{
  const std::string header = "{}";
  const auto path = std::filesystem::temp_directory_path() / "flowedge-no-profile.safetensors";
  path_ = path;
  std::ofstream output{path_, std::ios::binary | std::ios::trunc};
  ASSERT_TRUE(output.good());
  const std::uint64_t size = header.size();
  output.write(reinterpret_cast<const char*>(&size), sizeof(size));
  output.write(header.data(), static_cast<std::streamsize>(header.size()));
  output.close();
  const auto value =
      fe::safetensors_metadata_value(path_.string(), fe::kDeploymentProfileMetadataKey);
  ASSERT_TRUE(value.has_value());
  EXPECT_FALSE(value->has_value());
}

} // namespace
