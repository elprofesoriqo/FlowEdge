#include "deployment_profile.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace fe {
namespace {

struct Field
{
  std::string_view value{};
  bool present{false};
  bool string{false};
};

[[nodiscard]] std::string_view trim(std::string_view value) noexcept
{
  while (!value.empty() && (value.front() == ' ' || value.front() == '\n' ||
                            value.front() == '\r' || value.front() == '\t'))
    value.remove_prefix(1uz);
  while (!value.empty() && (value.back() == ' ' || value.back() == '\n' || value.back() == '\r' ||
                            value.back() == '\t'))
    value.remove_suffix(1uz);
  return value;
}

[[nodiscard]] Field field(std::string_view json, std::string_view key) noexcept
{
  const std::string needle = std::string{"\""} + std::string{key} + "\"";
  const std::size_t key_pos = json.find(needle);
  if (key_pos == std::string_view::npos)
    return {};
  const std::size_t colon = json.find(':', key_pos + needle.size());
  if (colon == std::string_view::npos)
    return {.present = true};

  std::size_t start = colon + 1uz;
  while (start < json.size() &&
         (json[start] == ' ' || json[start] == '\n' || json[start] == '\r' || json[start] == '\t'))
    ++start;
  if (start >= json.size())
    return {.present = true};
  if (json[start] == '"') {
    const std::size_t end = json.find('"', start + 1uz);
    if (end == std::string_view::npos)
      return {.present = true, .string = true};
    return {.value = json.substr(start + 1uz, end - start - 1uz), .present = true, .string = true};
  }
  const char opener = json[start];
  if (opener == '{' || opener == '[') {
    const char closer = opener == '{' ? '}' : ']';
    int depth{0};
    bool in_string{false};
    for (std::size_t index{start}; index < json.size(); ++index) {
      const char ch = json[index];
      if (ch == '"' && (index == 0uz || json[index - 1uz] != '\\'))
        in_string = !in_string;
      if (in_string)
        continue;
      if (ch == opener)
        ++depth;
      else if (ch == closer && --depth == 0)
        return {.value = json.substr(start, index - start + 1uz), .present = true};
    }
    return {.present = true};
  }
  std::size_t end = start;
  while (end < json.size() && json[end] != ',' && json[end] != '}')
    ++end;
  return {.value = trim(json.substr(start, end - start)), .present = true};
}

[[nodiscard]] bool parse_uint(std::string_view value, std::size_t& output) noexcept
{
  value = trim(value);
  if (value.empty())
    return false;
  std::uint64_t parsed{};
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (error != std::errc{} || end != value.data() + value.size() ||
      parsed > std::numeric_limits<std::size_t>::max())
    return false;
  output = static_cast<std::size_t>(parsed);
  return true;
}

[[nodiscard]] bool parse_float(std::string_view value, float& output) noexcept
{
  value = trim(value);
  if (value.empty())
    return false;
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), output);
  return error == std::errc{} && end == value.data() + value.size() && std::isfinite(output);
}

[[nodiscard]] bool parse_float_array(std::string_view value, std::vector<float>& output) noexcept
{
  value = trim(value);
  if (value.size() < 2uz || value.front() != '[' || value.back() != ']')
    return false;
  value = trim(value.substr(1uz, value.size() - 2uz));
  output.clear();
  if (value.empty())
    return true;
  while (!value.empty()) {
    const std::size_t comma = value.find(',');
    const std::string_view item = trim(value.substr(0uz, comma));
    float parsed{};
    if (!parse_float(item, parsed))
      return false;
    output.push_back(parsed);
    if (comma == std::string_view::npos)
      break;
    value = trim(value.substr(comma + 1uz));
    if (value.empty())
      return false;
  }
  return true;
}

[[nodiscard]] bool valid_hash(std::string_view value) noexcept
{
  if (value.size() != 71uz || value.substr(0uz, 7uz) != "sha256:")
    return false;
  return std::ranges::all_of(value.substr(7uz), [](const char ch) noexcept {
    return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F');
  });
}

[[nodiscard]] std::string_view action_units_name(DeploymentActionUnits units) noexcept
{
  return units == DeploymentActionUnits::kNormalized ? "normalized" : "physical";
}

[[nodiscard]] std::string_view normalization_name(DeploymentNormalization normalization) noexcept
{
  return normalization == DeploymentNormalization::kNone ? "none" : "minmax";
}

[[nodiscard]] std::string_view solver_name(DeploymentSolver solver) noexcept
{
  switch (solver) {
  case DeploymentSolver::kEuler:
    return "euler";
  case DeploymentSolver::kHeun:
    return "heun";
  case DeploymentSolver::kRK4:
    return "rk4";
  }
  return {};
}

void append_uint(std::string& output, std::size_t value)
{
  std::array<char, 32> buffer{};
  const auto [end, error] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
  if (error != std::errc{})
    return;
  output.append(buffer.data(), static_cast<std::size_t>(end - buffer.data()));
}

void append_float(std::string& output, float value)
{
  std::array<char, 64> buffer{};
  const auto [end, error] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                                          std::chars_format::general);
  if (error != std::errc{})
    return;
  output.append(buffer.data(), static_cast<std::size_t>(end - buffer.data()));
}

} // namespace

std::string_view deployment_profile_error_message(DeploymentProfileError error) noexcept
{
  switch (error) {
  case DeploymentProfileError::kInvalidJson:
    return "deployment profile JSON is invalid";
  case DeploymentProfileError::kMissingField:
    return "deployment profile field is missing";
  case DeploymentProfileError::kInvalidField:
    return "deployment profile field is invalid";
  case DeploymentProfileError::kUnsupportedProfileVersion:
    return "deployment profile version is unsupported";
  case DeploymentProfileError::kUnsupportedCompatibilityVersion:
    return "model compatibility version is unsupported";
  case DeploymentProfileError::kInvalidObservationSchemaHash:
    return "observation_schema_hash must be sha256 followed by 64 hex digits";
  case DeploymentProfileError::kInvalidActionDim:
    return "action_dim must be between 1 and 65536";
  case DeploymentProfileError::kInvalidActionHorizon:
    return "action_horizon must be between 1 and 4096";
  case DeploymentProfileError::kInvalidActionUnits:
    return "action_units must be normalized or physical";
  case DeploymentProfileError::kInvalidNormalizationType:
    return "normalization_type must be none or minmax";
  case DeploymentProfileError::kInvalidNormalizationParameters:
    return "normalization_parameters must contain finite min/max arrays matching action_dim";
  case DeploymentProfileError::kInvalidSolverDefault:
    return "solver_default must be euler, heun, or rk4";
  case DeploymentProfileError::kInvalidSolverRange:
    return "solver_min_steps and solver_max_steps must be between 1 and 4096";
  }
  return "deployment profile is invalid";
}

std::expected<void, DeploymentProfileError> validate_deployment_profile(
    const DeploymentProfile& profile) noexcept
{
  if (profile.profile_version != DeploymentProfile::kVersion)
    return std::unexpected(DeploymentProfileError::kUnsupportedProfileVersion);
  if (profile.model_compatibility_version != DeploymentProfile::kCompatibilityVersion)
    return std::unexpected(DeploymentProfileError::kUnsupportedCompatibilityVersion);
  if (!valid_hash(profile.observation_schema_hash_view()))
    return std::unexpected(DeploymentProfileError::kInvalidObservationSchemaHash);
  if (profile.action_dim == 0uz || profile.action_dim > DeploymentProfile::kMaxActionDim)
    return std::unexpected(DeploymentProfileError::kInvalidActionDim);
  if (profile.action_horizon == 0uz ||
      profile.action_horizon > DeploymentProfile::kMaxActionHorizon)
    return std::unexpected(DeploymentProfileError::kInvalidActionHorizon);
  if (profile.action_units != DeploymentActionUnits::kNormalized &&
      profile.action_units != DeploymentActionUnits::kPhysical)
    return std::unexpected(DeploymentProfileError::kInvalidActionUnits);
  if (profile.normalization_type != DeploymentNormalization::kNone &&
      profile.normalization_type != DeploymentNormalization::kMinMax)
    return std::unexpected(DeploymentProfileError::kInvalidNormalizationType);
  if (profile.normalization_type == DeploymentNormalization::kNone) {
    if (!profile.normalization_min.empty() || !profile.normalization_max.empty())
      return std::unexpected(DeploymentProfileError::kInvalidNormalizationParameters);
  } else {
    if (profile.normalization_min.size() != profile.action_dim ||
        profile.normalization_max.size() != profile.action_dim)
      return std::unexpected(DeploymentProfileError::kInvalidNormalizationParameters);
    for (std::size_t index{}; index < profile.action_dim; ++index)
      if (!std::isfinite(profile.normalization_min[index]) ||
          !std::isfinite(profile.normalization_max[index]) ||
          profile.normalization_min[index] >= profile.normalization_max[index])
        return std::unexpected(DeploymentProfileError::kInvalidNormalizationParameters);
  }
  if (solver_name(profile.solver_default).empty())
    return std::unexpected(DeploymentProfileError::kInvalidSolverDefault);
  if (profile.solver_min_steps == 0uz || profile.solver_max_steps == 0uz ||
      profile.solver_min_steps > profile.solver_max_steps ||
      profile.solver_max_steps > DeploymentProfile::kMaxSolverSteps)
    return std::unexpected(DeploymentProfileError::kInvalidSolverRange);
  return {};
}

std::expected<DeploymentProfile, DeploymentProfileError> parse_deployment_profile(
    std::string_view json)
{
  json = trim(json);
  if (json.size() < 2uz || json.front() != '{' || json.back() != '}')
    return std::unexpected(DeploymentProfileError::kInvalidJson);

  DeploymentProfile profile{};
  const Field profile_version = field(json, "profile_version");
  const Field compatibility = field(json, "model_compatibility_version");
  const Field schema_hash = field(json, "observation_schema_hash");
  const Field action_dim = field(json, "action_dim");
  const Field action_horizon = field(json, "action_horizon");
  const Field action_units = field(json, "action_units");
  const Field normalization_type = field(json, "normalization_type");
  const Field normalization_parameters = field(json, "normalization_parameters");
  const Field solver_default = field(json, "solver_default");
  const Field solver_min = field(json, "solver_min_steps");
  const Field solver_max = field(json, "solver_max_steps");
  if (!profile_version.present || !compatibility.present || !schema_hash.present ||
      !action_dim.present || !action_horizon.present || !action_units.present ||
      !normalization_type.present || !normalization_parameters.present || !solver_default.present ||
      !solver_min.present || !solver_max.present)
    return std::unexpected(DeploymentProfileError::kMissingField);

  std::size_t value{};
  if (!parse_uint(profile_version.value, value) ||
      value > std::numeric_limits<std::uint32_t>::max())
    return std::unexpected(DeploymentProfileError::kInvalidField);
  profile.profile_version = static_cast<std::uint32_t>(value);
  if (!parse_uint(compatibility.value, value) || value > std::numeric_limits<std::uint32_t>::max())
    return std::unexpected(DeploymentProfileError::kInvalidField);
  profile.model_compatibility_version = static_cast<std::uint32_t>(value);
  if (!schema_hash.string || !valid_hash(schema_hash.value))
    return std::unexpected(DeploymentProfileError::kInvalidObservationSchemaHash);
  std::copy(schema_hash.value.begin(), schema_hash.value.end(),
            profile.observation_schema_hash.begin());
  profile.observation_schema_hash[schema_hash.value.size()] = '\0';
  if (!parse_uint(action_dim.value, profile.action_dim))
    return std::unexpected(DeploymentProfileError::kInvalidActionDim);
  if (!parse_uint(action_horizon.value, profile.action_horizon))
    return std::unexpected(DeploymentProfileError::kInvalidActionHorizon);
  if (!action_units.string)
    return std::unexpected(DeploymentProfileError::kInvalidActionUnits);
  if (action_units.value == "normalized")
    profile.action_units = DeploymentActionUnits::kNormalized;
  else if (action_units.value == "physical")
    profile.action_units = DeploymentActionUnits::kPhysical;
  else
    return std::unexpected(DeploymentProfileError::kInvalidActionUnits);
  if (!normalization_type.string)
    return std::unexpected(DeploymentProfileError::kInvalidNormalizationType);
  if (normalization_type.value == "none")
    profile.normalization_type = DeploymentNormalization::kNone;
  else if (normalization_type.value == "minmax")
    profile.normalization_type = DeploymentNormalization::kMinMax;
  else
    return std::unexpected(DeploymentProfileError::kInvalidNormalizationType);
  if (normalization_parameters.value.empty())
    return std::unexpected(DeploymentProfileError::kInvalidNormalizationParameters);
  const Field min_values = field(normalization_parameters.value, "min");
  const Field max_values = field(normalization_parameters.value, "max");
  if (profile.normalization_type == DeploymentNormalization::kMinMax) {
    if (!min_values.present || !max_values.present ||
        !parse_float_array(min_values.value, profile.normalization_min) ||
        !parse_float_array(max_values.value, profile.normalization_max))
      return std::unexpected(DeploymentProfileError::kInvalidNormalizationParameters);
  } else if (min_values.present || max_values.present) {
    return std::unexpected(DeploymentProfileError::kInvalidNormalizationParameters);
  }
  if (!solver_default.string)
    return std::unexpected(DeploymentProfileError::kInvalidSolverDefault);
  if (solver_default.value == "euler")
    profile.solver_default = DeploymentSolver::kEuler;
  else if (solver_default.value == "heun")
    profile.solver_default = DeploymentSolver::kHeun;
  else if (solver_default.value == "rk4")
    profile.solver_default = DeploymentSolver::kRK4;
  else
    return std::unexpected(DeploymentProfileError::kInvalidSolverDefault);
  if (!parse_uint(solver_min.value, profile.solver_min_steps) ||
      !parse_uint(solver_max.value, profile.solver_max_steps))
    return std::unexpected(DeploymentProfileError::kInvalidSolverRange);
  if (profile.profile_version != DeploymentProfile::kVersion)
    return std::unexpected(DeploymentProfileError::kUnsupportedProfileVersion);
  if (profile.model_compatibility_version != DeploymentProfile::kCompatibilityVersion)
    return std::unexpected(DeploymentProfileError::kUnsupportedCompatibilityVersion);
  if (const auto valid = validate_deployment_profile(profile); !valid)
    return std::unexpected(valid.error());
  return profile;
}

std::expected<std::string, DeploymentProfileError> serialize_deployment_profile(
    const DeploymentProfile& profile)
{
  if (const auto valid = validate_deployment_profile(profile); !valid)
    return std::unexpected(valid.error());

  std::string output;
  output.reserve(256uz + profile.action_dim * 24uz);
  output += "{\"profile_version\":";
  append_uint(output, profile.profile_version);
  output += ",\"model_compatibility_version\":";
  append_uint(output, profile.model_compatibility_version);
  output += ",\"observation_schema_hash\":\"";
  output += profile.observation_schema_hash_view();
  output += "\",\"action_dim\":";
  append_uint(output, profile.action_dim);
  output += ",\"action_horizon\":";
  append_uint(output, profile.action_horizon);
  output += ",\"action_units\":\"";
  output += action_units_name(profile.action_units);
  output += "\",\"normalization_type\":\"";
  output += normalization_name(profile.normalization_type);
  output += "\",\"normalization_parameters\":{";
  if (profile.normalization_type == DeploymentNormalization::kMinMax) {
    output += "\"min\":[";
    for (std::size_t index{}; index < profile.normalization_min.size(); ++index) {
      if (index != 0uz)
        output += ',';
      append_float(output, profile.normalization_min[index]);
    }
    output += "],\"max\":[";
    for (std::size_t index{}; index < profile.normalization_max.size(); ++index) {
      if (index != 0uz)
        output += ',';
      append_float(output, profile.normalization_max[index]);
    }
    output += ']';
  }
  output += "},\"solver_default\":\"";
  output += solver_name(profile.solver_default);
  output += "\",\"solver_min_steps\":";
  append_uint(output, profile.solver_min_steps);
  output += ",\"solver_max_steps\":";
  append_uint(output, profile.solver_max_steps);
  output += '}';
  return output;
}

} // namespace fe
