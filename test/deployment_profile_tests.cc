#include "protocol/deployment_profile.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <string>

namespace {

fe::DeploymentProfile valid_profile()
{
  fe::DeploymentProfile profile{};
  profile.observation_schema_hash.fill('\0');
  const std::string hash =
      "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  std::copy(hash.begin(), hash.end(), profile.observation_schema_hash.begin());
  profile.action_dim = 2uz;
  profile.action_horizon = 16uz;
  profile.normalization_type = fe::DeploymentNormalization::kMinMax;
  profile.normalization_min = {-1.0F, 0.0F};
  profile.normalization_max = {1.0F, 2.0F};
  profile.solver_default = fe::DeploymentSolver::kHeun;
  profile.solver_min_steps = 6uz;
  profile.solver_max_steps = 16uz;
  return profile;
}

TEST(DeploymentProfile, SerializesAndRoundTrips)
{
  const fe::DeploymentProfile expected = valid_profile();
  const auto encoded = fe::serialize_deployment_profile(expected);
  ASSERT_TRUE(encoded.has_value());

  const auto decoded = fe::parse_deployment_profile(*encoded);
  ASSERT_TRUE(decoded.has_value()) << fe::deployment_profile_error_message(decoded.error());
  EXPECT_EQ(*decoded, expected);
}

TEST(DeploymentProfile, RejectsMissingField)
{
  const auto parsed = fe::parse_deployment_profile("{\"profile_version\":1}");
  ASSERT_FALSE(parsed.has_value());
  EXPECT_EQ(parsed.error(), fe::DeploymentProfileError::kMissingField);
}

TEST(DeploymentProfile, RejectsUnsupportedVersion)
{
  fe::DeploymentProfile profile = valid_profile();
  profile.profile_version = 2u;
  const auto encoded = fe::serialize_deployment_profile(profile);
  ASSERT_FALSE(encoded.has_value());
  EXPECT_EQ(encoded.error(), fe::DeploymentProfileError::kUnsupportedProfileVersion);
}

TEST(DeploymentProfile, RejectsInvalidActionDimensions)
{
  fe::DeploymentProfile profile = valid_profile();
  profile.action_dim = 0uz;
  const auto result = fe::validate_deployment_profile(profile);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), fe::DeploymentProfileError::kInvalidActionDim);
}

TEST(DeploymentProfile, RejectsMismatchedNormalizationParameters)
{
  fe::DeploymentProfile profile = valid_profile();
  profile.normalization_max.pop_back();
  const auto result = fe::validate_deployment_profile(profile);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), fe::DeploymentProfileError::kInvalidNormalizationParameters);
}

TEST(DeploymentProfile, AllowsNoNormalizationParameters)
{
  fe::DeploymentProfile profile = valid_profile();
  profile.normalization_type = fe::DeploymentNormalization::kNone;
  profile.normalization_min.clear();
  profile.normalization_max.clear();
  const auto encoded = fe::serialize_deployment_profile(profile);
  ASSERT_TRUE(encoded.has_value());
  const auto decoded = fe::parse_deployment_profile(*encoded);
  ASSERT_TRUE(decoded.has_value());
  EXPECT_EQ(decoded->normalization_type, fe::DeploymentNormalization::kNone);
}

} // namespace
