#include "loader/safetensors.h"
#include "protocol/model_identity.h"
#include "runtime/engine_runtime.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kMaxTensors = 4096uz;

struct Report
{
  std::string family{"unknown"};
  std::uint32_t precision{FE_PRECISION_UNKNOWN};
  std::size_t weights_bytes{};
  std::size_t arena_bytes{};
  std::size_t persistent_state_bytes{};
  std::size_t d_model{};
  std::size_t n_layers{};
  std::size_t d_inner{};
  std::size_t d_state{};
  std::size_t d_conv{};
  std::size_t action_dim{};
  std::size_t condition_dim{};
  std::size_t action_horizon{};
  std::size_t action_steps{};
  std::size_t observation_steps{};
  std::size_t diffusion_stages{};
  std::size_t diffusion_kernel{};
  std::size_t diffusion_groups{};
  std::size_t timestep_dim{};
  std::size_t train_timesteps{};
  bool clip_sample{};
  float clip_sample_range{};
  std::array<std::uint8_t, FE_MODEL_DIGEST_BYTES> digest{};
  std::size_t tensor_count{};
  std::vector<std::string> unsupported{};
  std::vector<std::string> missing{};
  std::vector<std::string> errors{};
};

const fe::TensorMetadata* find_tensor(std::span<const fe::TensorMetadata> tensors,
                                      std::string_view name) noexcept
{
  for (const auto& tensor : tensors)
    if (tensor.name_view() == name)
      return &tensor;
  return nullptr;
}

bool has_prefix(std::span<const fe::TensorMetadata> tensors, std::string_view prefix) noexcept
{
  for (const auto& tensor : tensors)
    if (tensor.name_view().starts_with(prefix))
      return true;
  return false;
}

std::size_t layer_count(std::span<const fe::TensorMetadata> tensors) noexcept
{
  constexpr std::string_view prefix = "backbone.layers.";
  std::size_t count{0uz};
  for (const auto& tensor : tensors) {
    const std::string_view name = tensor.name_view();
    if (!name.starts_with(prefix))
      continue;
    const auto begin = name.data() + prefix.size();
    const auto end = name.data() + name.size();
    std::size_t index{};
    const auto [next, error] = std::from_chars(begin, end, index);
    if (error == std::errc{} && next < end && *next == '.' && index < 1024uz)
      count = std::max(count, index + 1uz);
  }
  return count;
}

void missing_if_absent(std::span<const fe::TensorMetadata> tensors, std::string_view name,
                       Report& report)
{
  if (find_tensor(tensors, name) == nullptr)
    report.missing.emplace_back(name);
}

bool shape_is(const fe::TensorMetadata* tensor, std::initializer_list<std::size_t> expected)
{
  if (tensor == nullptr || tensor->ndim != expected.size())
    return false;
  std::size_t index{0uz};
  for (const std::size_t value : expected)
    if (tensor->shape[index++] != value)
      return false;
  return true;
}

void inspect_mamba(std::span<const fe::TensorMetadata> tensors, Report& report)
{
  const auto* embedding = find_tensor(tensors, "backbone.embeddings.weight");
  const auto* a_log = find_tensor(tensors, "backbone.layers.0.mixer.A_log");
  const auto* conv = find_tensor(tensors, "backbone.layers.0.mixer.conv1d.weight");
  const auto* x_proj = find_tensor(tensors, "backbone.layers.0.mixer.x_proj.weight");
  const auto* norm = find_tensor(tensors, "backbone.norm_f.weight");
  for (const std::string_view name :
       {"backbone.embeddings.weight", "backbone.layers.0.mixer.A_log",
        "backbone.layers.0.mixer.conv1d.weight", "backbone.layers.0.mixer.x_proj.weight",
        "backbone.norm_f.weight"})
    missing_if_absent(tensors, name, report);
  if (embedding == nullptr || a_log == nullptr || conv == nullptr || x_proj == nullptr ||
      norm == nullptr)
    return;

  report.d_model = embedding->ndim == 2uz ? embedding->shape[1] : 0uz;
  report.d_inner = a_log->ndim == 2uz ? a_log->shape[0] : 0uz;
  report.d_state = a_log->ndim == 2uz ? a_log->shape[1] : 0uz;
  report.d_conv = conv->ndim == 3uz ? conv->shape[2] : 0uz;
  report.n_layers = layer_count(tensors);
  if (!shape_is(norm, {report.d_model}))
    report.errors.emplace_back("backbone.norm_f.weight has an incompatible shape");
  if (report.n_layers == 0uz)
    report.errors.emplace_back("backbone has no numbered layers");
  if (report.d_inner == 0uz || report.d_state == 0uz || report.d_conv == 0uz)
    report.errors.emplace_back("backbone dimensions are zero or malformed");

  for (std::size_t layer{0uz}; layer < report.n_layers; ++layer) {
    const std::string prefix = "backbone.layers." + std::to_string(layer) + ".";
    const std::array<std::string, 10> names = {prefix + "norm.weight",
                                               prefix + "mixer.in_proj.weight",
                                               prefix + "mixer.conv1d.weight",
                                               prefix + "mixer.conv1d.bias",
                                               prefix + "mixer.x_proj.weight",
                                               prefix + "mixer.dt_proj.weight",
                                               prefix + "mixer.dt_proj.bias",
                                               prefix + "mixer.A_log",
                                               prefix + "mixer.D",
                                               prefix + "mixer.out_proj.weight"};
    for (const std::string& name : names)
      missing_if_absent(tensors, name, report);
  }
  if (!shape_is(embedding, {embedding->shape[0], report.d_model}) ||
      !shape_is(a_log, {report.d_inner, report.d_state}) ||
      !shape_is(conv, {report.d_inner, 1uz, report.d_conv}) || x_proj->ndim != 2uz)
    report.errors.emplace_back("backbone root tensor shapes are incompatible");
  if (report.d_inner != 0uz && report.d_state != 0uz && x_proj->ndim == 2uz &&
      x_proj->shape[0] <= 2uz * report.d_state)
    report.errors.emplace_back("backbone x_proj has no positive dt rank");
  if (report.d_inner != 0uz && report.d_state != 0uz && report.d_conv != 0uz)
    report.persistent_state_bytes =
        report.n_layers * report.d_inner * (report.d_conv + report.d_state) * sizeof(float);
}

void inspect_flow(std::span<const fe::TensorMetadata> tensors, Report& report)
{
  const auto* input = find_tensor(tensors, "flow.in_proj.weight");
  const auto* time = find_tensor(tensors, "flow.time_proj.weight");
  const auto* condition = find_tensor(tensors, "flow.cond_proj.weight");
  const auto* output = find_tensor(tensors, "flow.out_proj.weight");
  for (const std::string_view name : {"flow.in_proj.weight", "flow.time_proj.weight",
                                      "flow.cond_proj.weight", "flow.out_proj.weight"})
    missing_if_absent(tensors, name, report);
  if (input == nullptr || time == nullptr || condition == nullptr || output == nullptr)
    return;
  if (input->ndim != 2uz || time->ndim != 2uz || condition->ndim != 2uz || output->ndim != 2uz)
    report.errors.emplace_back("flow root tensor shapes must be matrices");
  else {
    const std::size_t hidden = input->shape[0];
    report.action_dim = input->shape[1];
    report.condition_dim = condition->shape[1];
    if (time->shape[0] != hidden || output->shape[0] != report.action_dim ||
        output->shape[1] != hidden || condition->shape[0] != hidden)
      report.errors.emplace_back("flow root tensor shapes are incompatible");
    if (time->shape[1] == 0uz || (time->shape[1] % 2uz) != 0uz)
      report.errors.emplace_back("flow time projection width must be positive and even");
  }
}

void inspect_diffusion(std::span<const fe::TensorMetadata> tensors, Report& report)
{
  for (const std::string_view name :
       {"dp.meta", "dp.dims", "dp.action_min", "dp.action_max", "dp.te1.w", "dp.te1.b", "dp.te2.w",
        "dp.te2.b", "dp.f.c.w", "dp.f.c.b", "dp.f.n.w", "dp.f.n.b", "dp.f.o.w", "dp.f.o.b"})
    missing_if_absent(tensors, name, report);

  const auto* meta = find_tensor(tensors, "dp.meta");
  const auto* dims = find_tensor(tensors, "dp.dims");
  const auto* action_min = find_tensor(tensors, "dp.action_min");
  const auto* action_max = find_tensor(tensors, "dp.action_max");
  if (meta != nullptr && !shape_is(meta, {16uz}))
    report.errors.emplace_back("dp.meta must contain exactly 16 values");
  if (dims != nullptr && (dims->ndim != 1uz || dims->shape[0] == 0uz))
    report.errors.emplace_back("dp.dims must be a non-empty vector");
  if (action_min != nullptr && action_max != nullptr &&
      (action_min->ndim != 1uz || action_max->ndim != 1uz ||
       action_min->shape[0] != action_max->shape[0] || action_min->shape[0] == 0uz))
    report.errors.emplace_back(
        "Diffusion action normalization vectors must have equal positive length");
}

void record_diffusion_config(const fe::DiffusionConfig& config, Report& report) noexcept
{
  report.action_dim = config.action_dim;
  report.condition_dim = config.condition_dim;
  report.action_horizon = config.horizon;
  report.action_steps = config.action_steps;
  report.observation_steps = config.observation_steps;
  report.diffusion_stages = config.stages;
  report.diffusion_kernel = config.kernel;
  report.diffusion_groups = config.groups;
  report.timestep_dim = config.timestep_dim;
  report.train_timesteps = config.train_timesteps;
  report.clip_sample = config.clip_sample;
  report.clip_sample_range = config.clip_sample_range;
}

const char* precision_name(std::uint32_t precision) noexcept
{
  switch (precision) {
  case FE_PRECISION_F32:
    return "f32";
  case FE_PRECISION_BF16:
    return "bf16";
  case FE_PRECISION_MIXED:
    return "mixed";
  default:
    return "unknown";
  }
}

void json_string(std::string_view value)
{
  std::cout << '"';
  for (const char c : value) {
    if (c == '"' || c == '\\')
      std::cout << '\\';
    std::cout << c;
  }
  std::cout << '"';
}

void json_strings(const std::vector<std::string>& values)
{
  std::cout << '[';
  for (std::size_t i{0uz}; i < values.size(); ++i) {
    if (i != 0uz)
      std::cout << ',';
    json_string(values[i]);
  }
  std::cout << ']';
}

void print_human(const Report& report)
{
  std::cout << "Model family        " << report.family << '\n'
            << "Precision           " << precision_name(report.precision) << '\n'
            << "Tensors             " << report.tensor_count << '\n'
            << "\nBackbone\n"
            << "  layers            " << report.n_layers << '\n'
            << "  d_model           " << report.d_model << '\n'
            << "  d_inner           " << report.d_inner << '\n'
            << "  d_state           " << report.d_state << '\n'
            << "  d_conv            " << report.d_conv << '\n'
            << "\nAction head\n"
            << "  action_dim        " << report.action_dim << '\n'
            << "  condition_dim     " << report.condition_dim << '\n';
  if (report.family == "diffusion-policy") {
    std::cout << "\nDiffusion Policy\n"
              << "  action_horizon    " << report.action_horizon << '\n'
              << "  action_steps      " << report.action_steps << '\n'
              << "  observation_steps " << report.observation_steps << '\n'
              << "  stages            " << report.diffusion_stages << '\n'
              << "  kernel            " << report.diffusion_kernel << '\n'
              << "  groups            " << report.diffusion_groups << '\n'
              << "  timestep_dim      " << report.timestep_dim << '\n'
              << "  train_timesteps  " << report.train_timesteps << '\n'
              << "  clip_sample       " << (report.clip_sample ? "true" : "false") << '\n'
              << "  clip_range        " << report.clip_sample_range << '\n';
  }
  std::cout << "\nMemory\n"
            << "  weights           " << report.weights_bytes << " bytes\n"
            << "  arena             " << report.arena_bytes << " bytes\n"
            << "  persistent_state  " << report.persistent_state_bytes << " bytes\n"
            << "\nCompatibility       " << (report.errors.empty() ? "supported" : "unsupported")
            << '\n';
  for (const auto& error : report.errors)
    std::cout << "ERROR: " << error << '\n';
  for (const auto& name : report.missing)
    std::cout << "Missing tensor      " << name << '\n';
  for (const auto& name : report.unsupported)
    std::cout << "Unsupported tensor  " << name << '\n';
}

void print_json(const Report& report)
{
  std::cout << "{\"schema_version\":1,\"checkpoint\":{";
  std::cout << "\"tensor_count\":" << report.tensor_count << ",\"precision\":";
  json_string(precision_name(report.precision));
  std::cout << ",\"digest\":";
  std::string digest;
  digest.reserve(FE_MODEL_DIGEST_BYTES * 2uz);
  constexpr char hex[] = "0123456789abcdef";
  for (const std::uint8_t byte : report.digest) {
    digest.push_back(hex[byte >> 4u]);
    digest.push_back(hex[byte & 0x0fu]);
  }
  json_string(digest);
  std::cout << "},\"model\":{";
  std::cout << "\"family\":";
  json_string(report.family);
  std::cout << ",\"d_model\":" << report.d_model << ",\"layers\":" << report.n_layers
            << ",\"d_inner\":" << report.d_inner << ",\"d_state\":" << report.d_state
            << ",\"d_conv\":" << report.d_conv << ",\"action_dim\":" << report.action_dim
            << ",\"condition_dim\":" << report.condition_dim
            << ",\"action_horizon\":" << report.action_horizon
            << ",\"action_steps\":" << report.action_steps
            << ",\"observation_steps\":" << report.observation_steps
            << ",\"diffusion_stages\":" << report.diffusion_stages
            << ",\"diffusion_kernel\":" << report.diffusion_kernel
            << ",\"diffusion_groups\":" << report.diffusion_groups
            << ",\"timestep_dim\":" << report.timestep_dim
            << ",\"train_timesteps\":" << report.train_timesteps
            << ",\"clip_sample\":" << (report.clip_sample ? "true" : "false")
            << ",\"clip_sample_range\":" << report.clip_sample_range << "},\"memory\":{"
            << "\"weights_bytes\":" << report.weights_bytes
            << ",\"arena_bytes\":" << report.arena_bytes
            << ",\"persistent_state_bytes\":" << report.persistent_state_bytes
            << "},\"compatibility\":{";
  std::cout << "\"supported\":" << (report.errors.empty() ? "true" : "false") << ",\"errors\":";
  json_strings(report.errors);
  std::cout << ",\"missing_required_tensors\":";
  json_strings(report.missing);
  std::cout << ",\"unsupported_tensors\":";
  json_strings(report.unsupported);
  std::cout << "}}\n";
}

void usage()
{
  std::cout << "usage: flowedge-inspect <model.safetensors> [--json]\n";
}

} // namespace

int main(int argc, char** argv)
{
  if (argc < 2 || std::string_view{argv[1]} == "--help" || std::string_view{argv[1]} == "-h") {
    usage();
    return argc < 2 ? 2 : 0;
  }
  const std::string_view path{argv[1]};
  const bool json = argc == 3 && std::string_view{argv[2]} == "--json";
  if (argc > 3 || (argc == 3 && !json)) {
    usage();
    return 2;
  }

  std::array<fe::TensorMetadata, kMaxTensors> storage{};
  std::size_t tensor_count{}, total_bytes{}, unsupported_count{};
  const auto inspected =
      fe::inspect_safetensors(path, storage, tensor_count, total_bytes, unsupported_count);
  if (!inspected) {
    std::cerr << "error: " << inspected.error() << '\n';
    return 2;
  }
  const std::span<const fe::TensorMetadata> tensors{storage.data(), tensor_count};
  Report report{};
  report.tensor_count = tensor_count;
  report.weights_bytes = total_bytes;
  report.unsupported.reserve(unsupported_count);
  for (const auto& tensor : tensors)
    if (!tensor.supported())
      report.unsupported.emplace_back(tensor.name_view());
  const bool mamba = has_prefix(tensors, "backbone.");
  const bool flow = has_prefix(tensors, "flow.");
  const bool diffusion = has_prefix(tensors, "dp.");
  report.family = diffusion       ? "diffusion-policy"
                  : mamba && flow ? "mamba-flow"
                  : mamba         ? "mamba"
                  : flow          ? "flow-head"
                                  : "unknown";
  if (mamba)
    inspect_mamba(tensors, report);
  if (flow)
    inspect_flow(tensors, report);
  if (diffusion)
    inspect_diffusion(tensors, report);
  if (!mamba && !flow && !diffusion)
    report.errors.emplace_back("unknown model family");
  if (diffusion && (mamba || flow))
    report.errors.emplace_back(
        "Diffusion Policy tensors cannot be mixed with backbone or flow tensors");
  if (!report.unsupported.empty())
    report.errors.emplace_back("checkpoint contains unsupported tensor dtypes");
  if (mamba && flow && report.condition_dim != report.d_model)
    report.errors.emplace_back("flow condition_dim does not match backbone d_model");

  const auto weights = fe::ModelWeights::open(path);
  if (weights) {
    report.precision = fe::checkpoint_precision((*weights)->tensors());
    report.arena_bytes = fe::EngineRuntime::required_slab_bytes(**weights);
    const auto digest = fe::fingerprint_tensors((*weights)->tensors());
    report.digest = digest;
    if (diffusion) {
      if (report.arena_bytes == 0uz) {
        report.errors.emplace_back("Diffusion Policy metadata or tensor shapes are invalid");
      } else {
        const char* runtime_error = nullptr;
        fe::EngineRuntime runtime{*weights, report.arena_bytes, 0u, runtime_error};
        if (!runtime.valid())
          report.errors.emplace_back(runtime_error != nullptr
                                         ? runtime_error
                                         : "Diffusion Policy runtime initialization failed");
        else if (const auto* config = runtime.diffusion_config())
          record_diffusion_config(*config, report);
        else
          report.errors.emplace_back("Diffusion Policy runtime configuration is unavailable");
      }
    }
  } else {
    report.errors.emplace_back(weights.error());
  }
  if (json)
    print_json(report);
  else
    print_human(report);
  return report.errors.empty() ? 0 : 2;
}
