#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

namespace fe {

enum class DeploymentActionUnits : std::uint8_t
{
  kNormalized,
  kPhysical,
};

enum class DeploymentNormalization : std::uint8_t
{
  kNone,
  kMinMax,
};

enum class DeploymentSolver : std::uint8_t
{
  kEuler,
  kHeun,
  kRK4,
};

enum class DeploymentProfileError : std::uint8_t
{
  kInvalidJson,
  kMissingField,
  kInvalidField,
  kUnsupportedProfileVersion,
  kUnsupportedCompatibilityVersion,
  kInvalidObservationSchemaHash,
  kInvalidActionDim,
  kInvalidActionHorizon,
  kInvalidActionUnits,
  kInvalidNormalizationType,
  kInvalidNormalizationParameters,
  kInvalidSolverDefault,
  kInvalidSolverRange,
};

struct DeploymentProfile
{
  static constexpr std::uint32_t kVersion = 1u;
  static constexpr std::uint32_t kCompatibilityVersion = 1u;
  static constexpr std::size_t kObservationSchemaHashBytes = 72uz;
  static constexpr std::size_t kMaxActionDim = 65'536uz;
  static constexpr std::size_t kMaxActionHorizon = 4'096uz;
  static constexpr std::size_t kMaxSolverSteps = 4'096uz;

  std::uint32_t profile_version{kVersion};
  std::uint32_t model_compatibility_version{kCompatibilityVersion};
  std::array<char, kObservationSchemaHashBytes> observation_schema_hash{};
  std::size_t action_dim{};
  std::size_t action_horizon{};
  DeploymentActionUnits action_units{DeploymentActionUnits::kNormalized};
  DeploymentNormalization normalization_type{DeploymentNormalization::kNone};
  std::vector<float> normalization_min{};
  std::vector<float> normalization_max{};
  DeploymentSolver solver_default{DeploymentSolver::kEuler};
  std::size_t solver_min_steps{};
  std::size_t solver_max_steps{};

  [[nodiscard]] std::string_view observation_schema_hash_view() const noexcept
  {
    return {observation_schema_hash.data()};
  }

  [[nodiscard]] friend bool operator==(const DeploymentProfile&,
                                       const DeploymentProfile&) = default;
};

inline constexpr std::string_view kDeploymentProfileMetadataKey = "flowedge.deployment_profile";

[[nodiscard]] std::string_view deployment_profile_error_message(
    DeploymentProfileError error) noexcept;

[[nodiscard]] std::expected<void, DeploymentProfileError> validate_deployment_profile(
    const DeploymentProfile& profile) noexcept;

[[nodiscard]] std::expected<DeploymentProfile, DeploymentProfileError> parse_deployment_profile(
    std::string_view json);

[[nodiscard]] std::expected<std::string, DeploymentProfileError> serialize_deployment_profile(
    const DeploymentProfile& profile);

} // namespace fe
