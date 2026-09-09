#include "heads/diffusion/diffusion.h"

#include "kernels/kernels.h"
#include "loader/tensor_key.h"
#include "loader/weight_ops.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numbers>
#include <span>
#include <string_view>

namespace fe {
namespace {

constexpr std::size_t kMetaValues = 16uz;
constexpr std::size_t kSchemaVersion = 1uz;
constexpr std::size_t kFinalNormGroups = 8uz;
constexpr std::size_t kAlignmentFloats = kSimdAlign / sizeof(float);

[[nodiscard]] bool checked_add(std::size_t lhs, std::size_t rhs, std::size_t& out) noexcept
{
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
    return false;
  out = lhs + rhs;
  return true;
}

[[nodiscard]] bool checked_mul(std::size_t lhs, std::size_t rhs, std::size_t& out) noexcept
{
  if (lhs != 0uz && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    return false;
  out = lhs * rhs;
  return true;
}

[[nodiscard]] bool size_value(float value, std::size_t& out) noexcept
{
  if (!std::isfinite(value) || value < 0.0F || std::floor(value) != value ||
      value > static_cast<float>(std::numeric_limits<std::uint32_t>::max()))
    return false;
  out = static_cast<std::size_t>(value);
  return true;
}

[[nodiscard]] bool vector_f32(const TensorView* tensor, std::size_t size) noexcept
{
  return tensor != nullptr && tensor->is_f32() && tensor->ndim == 1u && tensor->shape[0] == size &&
         size <= std::numeric_limits<std::size_t>::max() / sizeof(float) &&
         tensor->bytes == size * sizeof(float);
}

[[nodiscard]] bool exact_storage(const TensorView* tensor, std::size_t elements,
                                 std::size_t element_bytes) noexcept
{
  return tensor != nullptr && elements <= std::numeric_limits<std::size_t>::max() / element_bytes &&
         tensor->bytes == elements * element_bytes;
}

class WorkspacePlan
{
public:
  WorkspacePlan() noexcept : current_{kAlignmentFloats - 1uz}, peak_{current_} {}

  [[nodiscard]] bool add(std::size_t values) noexcept
  {
    const std::size_t padding =
        (kAlignmentFloats - (current_ % kAlignmentFloats)) % kAlignmentFloats;
    if (!checked_add(current_, padding, current_) || !checked_add(current_, values, current_))
      return false;
    peak_ = std::max(peak_, current_);
    return true;
  }

  [[nodiscard]] bool add_product(std::size_t lhs, std::size_t rhs) noexcept
  {
    std::size_t values{0uz};
    return checked_mul(lhs, rhs, values) && add(values);
  }

  [[nodiscard]] bool temporary(std::initializer_list<std::size_t> allocations) noexcept
  {
    const std::size_t saved = current_;
    for (const std::size_t values : allocations)
      if (!add(values))
        return false;
    current_ = saved;
    return true;
  }

  [[nodiscard]] std::size_t peak() const noexcept { return peak_; }

private:
  std::size_t current_{};
  std::size_t peak_{};
};

[[nodiscard]] std::string_view suffix_key(std::span<char> buffer, std::string_view prefix,
                                          std::string_view suffix) noexcept
{
  TensorKeyBuilder key{buffer};
  if (!key.append(prefix) || !key.append(suffix))
    return {};
  return key.view();
}

[[nodiscard]] std::string_view indexed_prefix(std::span<char> buffer, std::string_view base,
                                              std::size_t index,
                                              std::string_view suffix = {}) noexcept
{
  TensorKeyBuilder key{buffer};
  if (!key.append(base) || !key.append(index) || !key.append(suffix))
    return {};
  return key.view();
}

void add_bias(std::span<float> values, std::span<const float> bias, std::size_t rows,
              std::size_t columns) noexcept
{
  for (std::size_t row{0uz}; row < rows; ++row)
    for (std::size_t column{0uz}; column < columns; ++column)
      values[(row * columns) + column] += bias[column];
}

[[nodiscard]] double alpha_bar(double time) noexcept
{
  const double phase = ((time + 0.008) / 1.008) * (std::numbers::pi / 2.0);
  const double cosine = std::cos(phase);
  return cosine * cosine;
}

class GaussianGenerator
{
public:
  explicit GaussianGenerator(std::uint64_t seed) noexcept : state_{seed + 0x9e3779b97f4a7c15ULL} {}

  [[nodiscard]] float next() noexcept
  {
    if (has_spare_) {
      has_spare_ = false;
      return spare_;
    }
    const float first = std::max(uniform(), std::numeric_limits<float>::min());
    const float second = uniform();
    const float radius = std::sqrt(-2.0F * std::log(first));
    const float angle = 2.0F * std::numbers::pi_v<float> * second;
    spare_ = radius * std::sin(angle);
    has_spare_ = true;
    return radius * std::cos(angle);
  }

private:
  [[nodiscard]] std::uint64_t integer() noexcept
  {
    state_ += 0x9e3779b97f4a7c15ULL;
    std::uint64_t value = state_;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31u);
  }

  [[nodiscard]] float uniform() noexcept
  {
    return static_cast<float>((integer() >> 40u) + 1u) / 16777217.0F;
  }

  std::uint64_t state_{};
  float spare_{};
  bool has_spare_{};
};

} // namespace

std::size_t DiffusionHead::required_persistent_floats(std::span<const TensorView> weights) noexcept
{
  const TensorView* const meta = find_tensor(weights, "dp.meta");
  if (meta == nullptr || !meta->is_f32() || meta->ndim != 1u || meta->shape[0] != kMetaValues ||
      !exact_storage(meta, kMetaValues, sizeof(float)))
    return 0uz;
  std::size_t train_timesteps{0uz};
  return size_value(meta->as_f32()[10], train_timesteps) ? train_timesteps : 0uz;
}

std::size_t DiffusionHead::required_workspace_floats(std::span<const TensorView> weights) noexcept
{
  const TensorView* const meta = find_tensor(weights, "dp.meta");
  const TensorView* const dims = find_tensor(weights, "dp.dims");
  if (meta == nullptr || dims == nullptr || !meta->is_f32() || !dims->is_f32() ||
      meta->ndim != 1u || meta->shape[0] != kMetaValues || dims->ndim != 1u ||
      !exact_storage(meta, kMetaValues, sizeof(float)))
    return 0uz;
  const float* const values = meta->as_f32();
  std::size_t action_dim{0uz};
  std::size_t horizon{0uz};
  std::size_t condition_dim{0uz};
  std::size_t stages{0uz};
  std::size_t timestep_dim{0uz};
  if (!size_value(values[1], action_dim) || !size_value(values[2], horizon) ||
      !size_value(values[5], condition_dim) || !size_value(values[6], stages) ||
      !size_value(values[9], timestep_dim) || stages < 2uz || stages > kMaxStages ||
      dims->shape[0] != stages || !exact_storage(dims, stages, sizeof(float)))
    return 0uz;
  std::array<std::size_t, kMaxStages> stage_dims{};
  for (std::size_t i{0uz}; i < stages; ++i) {
    if (!size_value(dims->as_f32()[i], stage_dims[i]) || stage_dims[i] == 0uz)
      return 0uz;
  }
  std::size_t sample_values{0uz};
  if (!checked_mul(action_dim, horizon, sample_values))
    return 0uz;

  WorkspacePlan plan;
  std::size_t time_hidden{0uz};
  std::size_t condition_features{0uz};
  if (!checked_mul(timestep_dim, 4uz, time_hidden) ||
      !checked_add(timestep_dim, condition_dim, condition_features) || !plan.add(timestep_dim) ||
      !plan.add(time_hidden) || !plan.add(condition_features) || !plan.add(sample_values))
    return 0uz;
  const auto residual = [&plan](std::size_t channels, std::size_t length) noexcept {
    std::size_t activation{0uz};
    std::size_t modulation{0uz};
    return checked_mul(channels, length, activation) && checked_mul(2uz, channels, modulation) &&
           plan.add(activation) && plan.temporary({activation, modulation});
  };

  std::size_t length = horizon;
  for (std::size_t stage{0uz}; stage < stages; ++stage) {
    const std::size_t channels = stage_dims[stage];
    for (std::size_t block{0uz}; block < 2uz; ++block)
      if (!residual(channels, length))
        return 0uz;
    if (stage + 1uz < stages) {
      length /= 2uz;
      if (!plan.add_product(channels, length))
        return 0uz;
    }
  }

  const std::size_t bottom_channels = stage_dims[stages - 1uz];
  for (std::size_t block{0uz}; block < 2uz; ++block)
    if (!residual(bottom_channels, length))
      return 0uz;

  for (std::size_t stage{0uz}; stage + 1uz < stages; ++stage) {
    const std::size_t high = stage_dims[stages - 1uz - stage];
    const std::size_t low = stage_dims[stages - 2uz - stage];
    std::size_t concatenated_channels{0uz};
    if (!checked_mul(2uz, high, concatenated_channels) ||
        !plan.add_product(concatenated_channels, length) || !residual(low, length) ||
        !residual(low, length))
      return 0uz;
    if (!checked_mul(length, 2uz, length) || !plan.add_product(low, length))
      return 0uz;
  }
  if (!plan.add_product(stage_dims[0], horizon) || !plan.add(sample_values))
    return 0uz;

  std::size_t sampler_values{0uz};
  std::size_t total{0uz};
  return checked_mul(2uz, sample_values, sampler_values) &&
                 checked_add(sampler_values, plan.peak(), total)
             ? total
             : 0uz;
}

DiffusionHead::DiffusionHead(std::span<const TensorView> weights, Arena& persistent) noexcept
{
  const TensorView* const meta_tensor = find_tensor(weights, "dp.meta");
  const TensorView* const dims_tensor = find_tensor(weights, "dp.dims");
  if (meta_tensor == nullptr || dims_tensor == nullptr || !meta_tensor->is_f32() ||
      !dims_tensor->is_f32() || meta_tensor->ndim != 1u || meta_tensor->shape[0] != kMetaValues ||
      dims_tensor->ndim != 1u || !exact_storage(meta_tensor, kMetaValues, sizeof(float)))
    return;
  const float* const meta = meta_tensor->as_f32();
  std::size_t schema{0uz};
  std::size_t clip_sample{0uz};
  std::size_t beta_schedule{0uz};
  std::size_t prediction_type{0uz};
  std::size_t normalization{0uz};
  if (!size_value(meta[0], schema) || !size_value(meta[1], cfg_.action_dim) ||
      !size_value(meta[2], cfg_.horizon) || !size_value(meta[3], cfg_.action_steps) ||
      !size_value(meta[4], cfg_.observation_steps) || !size_value(meta[5], cfg_.condition_dim) ||
      !size_value(meta[6], cfg_.stages) || !size_value(meta[7], cfg_.kernel) ||
      !size_value(meta[8], cfg_.groups) || !size_value(meta[9], cfg_.timestep_dim) ||
      !size_value(meta[10], cfg_.train_timesteps) || !size_value(meta[11], beta_schedule) ||
      !size_value(meta[12], prediction_type) || !size_value(meta[13], clip_sample) ||
      !size_value(meta[15], normalization) || schema != kSchemaVersion || beta_schedule != 0uz ||
      prediction_type != 0uz || normalization != 0uz || clip_sample > 1uz ||
      cfg_.action_dim == 0uz || cfg_.horizon == 0uz || cfg_.action_steps == 0uz ||
      cfg_.observation_steps == 0uz || cfg_.observation_steps > cfg_.horizon ||
      cfg_.action_steps > cfg_.horizon - cfg_.observation_steps + 1uz ||
      cfg_.condition_dim == 0uz || cfg_.stages < 2uz || cfg_.stages > kMaxStages ||
      cfg_.kernel == 0uz || cfg_.kernel % 2uz == 0uz || cfg_.groups == 0uz ||
      cfg_.timestep_dim < 4uz || cfg_.timestep_dim % 2uz != 0uz || cfg_.train_timesteps == 0uz ||
      dims_tensor->shape[0] != cfg_.stages ||
      !exact_storage(dims_tensor, cfg_.stages, sizeof(float)) || !std::isfinite(meta[14]) ||
      meta[14] <= 0.0F)
    return;
  cfg_.clip_sample = clip_sample != 0uz;
  cfg_.clip_sample_range = meta[14];
  for (std::size_t stage{0uz}; stage < cfg_.stages; ++stage)
    if (!size_value(dims_tensor->as_f32()[stage], dims_[stage]) || dims_[stage] == 0uz ||
        dims_[stage] % cfg_.groups != 0uz)
      return;
  if (dims_[0] % kFinalNormGroups != 0uz)
    return;
  const std::size_t down_factor = 1uz << (cfg_.stages - 1uz);
  if (cfg_.horizon % down_factor != 0uz)
    return;

  const TensorView* const action_min = find_tensor(weights, "dp.action_min");
  const TensorView* const action_max = find_tensor(weights, "dp.action_max");
  if (!vector_f32(action_min, cfg_.action_dim) || !vector_f32(action_max, cfg_.action_dim))
    return;
  action_min_ = action_min->as_f32();
  action_max_ = action_max->as_f32();
  for (std::size_t i{0uz}; i < cfg_.action_dim; ++i)
    if (!std::isfinite(action_min_[i]) || !std::isfinite(action_max_[i]) ||
        action_max_[i] <= action_min_[i])
      return;

  const std::size_t timestep_hidden = cfg_.timestep_dim * 4uz;
  if (!load_linear(weights, "dp.te1", timestep_hidden, cfg_.timestep_dim, timestep_in_) ||
      !load_linear(weights, "dp.te2", cfg_.timestep_dim, timestep_hidden, timestep_out_))
    return;
  const std::size_t film_features = cfg_.timestep_dim + cfg_.condition_dim;

  for (std::size_t stage{0uz}; stage < cfg_.stages; ++stage) {
    const std::size_t in_channels = stage == 0uz ? cfg_.action_dim : dims_[stage - 1uz];
    const std::size_t out_channels = dims_[stage];
    for (std::size_t block{0uz}; block < 2uz; ++block) {
      std::array<char, 32> prefix_buffer{};
      TensorKeyBuilder prefix{prefix_buffer};
      if (!prefix.append("dp.d") || !prefix.append(stage) || !prefix.append(".r") ||
          !prefix.append(block) ||
          !load_residual(weights, prefix.view(), block == 0uz ? in_channels : out_channels,
                         out_channels, down_[stage].residuals[block]))
        return;
      if (down_[stage].residuals[block].film.in_features != film_features)
        return;
    }
    down_[stage].identity_downsample = stage + 1uz == cfg_.stages;
    if (!down_[stage].identity_downsample) {
      std::array<char, 24> prefix_buffer{};
      const std::string_view prefix = indexed_prefix(prefix_buffer, "dp.d", stage, ".ds");
      if (prefix.empty() ||
          !load_conv(weights, prefix, out_channels, out_channels, 3uz, down_[stage].downsample))
        return;
    }
  }

  for (std::size_t block{0uz}; block < 2uz; ++block) {
    std::array<char, 24> prefix_buffer{};
    const std::string_view prefix = indexed_prefix(prefix_buffer, "dp.m", block);
    if (prefix.empty() ||
        !load_residual(weights, prefix, dims_[cfg_.stages - 1uz], dims_[cfg_.stages - 1uz],
                       middle_[block]) ||
        middle_[block].film.in_features != film_features)
      return;
  }

  for (std::size_t stage{0uz}; stage + 1uz < cfg_.stages; ++stage) {
    const std::size_t high = dims_[cfg_.stages - 1uz - stage];
    const std::size_t low = dims_[cfg_.stages - 2uz - stage];
    for (std::size_t block{0uz}; block < 2uz; ++block) {
      std::array<char, 32> prefix_buffer{};
      TensorKeyBuilder prefix{prefix_buffer};
      if (!prefix.append("dp.u") || !prefix.append(stage) || !prefix.append(".r") ||
          !prefix.append(block) ||
          !load_residual(weights, prefix.view(), block == 0uz ? 2uz * high : low, low,
                         up_[stage].residuals[block]) ||
          up_[stage].residuals[block].film.in_features != film_features)
        return;
    }
    std::array<char, 24> prefix_buffer{};
    const std::string_view prefix = indexed_prefix(prefix_buffer, "dp.u", stage, ".us");
    if (prefix.empty() || !load_conv(weights, prefix, low, low, 4uz, up_[stage].upsample))
      return;
  }

  if (!load_conv(weights, "dp.f.c", dims_[0], dims_[0], cfg_.kernel, final_conv_) ||
      !load_norm(weights, "dp.f.n", dims_[0], final_norm_) ||
      !load_conv(weights, "dp.f.o", cfg_.action_dim, dims_[0], 1uz, output_conv_))
    return;

  auto* const alphas = persistent.alloc_array<float, kSimdAlign>(cfg_.train_timesteps);
  if (alphas == nullptr)
    return;
  float cumulative{1.0F};
  for (std::size_t step{0uz}; step < cfg_.train_timesteps; ++step) {
    const double begin = static_cast<double>(step) / static_cast<double>(cfg_.train_timesteps);
    const double end = static_cast<double>(step + 1uz) / static_cast<double>(cfg_.train_timesteps);
    const float beta =
        static_cast<float>(std::min(1.0 - (alpha_bar(end) / alpha_bar(begin)), 0.999));
    cumulative *= 1.0F - beta;
    alphas[step] = cumulative;
  }
  alphas_cumprod_ = alphas;
  workspace_floats_ = required_workspace_floats(weights);
  ok_ = workspace_floats_ != 0uz;
}

bool DiffusionHead::load_conv(std::span<const TensorView> weights, std::string_view prefix,
                              std::size_t out_channels, std::size_t in_channels, std::size_t kernel,
                              ConvWeights& result) noexcept
{
  std::array<char, 48> weight_key{};
  std::array<char, 48> bias_key{};
  const TensorView* const weight = find_tensor(weights, suffix_key(weight_key, prefix, ".w"));
  const TensorView* const bias = find_tensor(weights, suffix_key(bias_key, prefix, ".b"));
  std::size_t weight_elements{0uz};
  const bool valid_size = checked_mul(out_channels, in_channels, weight_elements) &&
                          checked_mul(weight_elements, kernel, weight_elements);
  if (weight == nullptr || !weight->is_f32() || weight->ndim != 3u ||
      weight->shape[0] != out_channels || weight->shape[1] != in_channels ||
      weight->shape[2] != kernel || !valid_size ||
      !exact_storage(weight, weight_elements, sizeof(float)) || !vector_f32(bias, out_channels))
    return false;
  result = {.weight = weight->as_f32(),
            .bias = bias->as_f32(),
            .in_channels = in_channels,
            .out_channels = out_channels,
            .kernel = kernel};
  return true;
}

bool DiffusionHead::load_linear(std::span<const TensorView> weights, std::string_view prefix,
                                std::size_t out_features, std::size_t in_features,
                                LinearWeights& result) noexcept
{
  std::array<char, 48> weight_key{};
  std::array<char, 48> bias_key{};
  const TensorView* const weight = find_tensor(weights, suffix_key(weight_key, prefix, ".w"));
  const TensorView* const bias = find_tensor(weights, suffix_key(bias_key, prefix, ".b"));
  std::size_t weight_elements{0uz};
  const std::size_t element_bytes =
      weight != nullptr && weight->is_bf16() ? sizeof(std::uint16_t) : sizeof(float);
  if (weight == nullptr || weight->ndim != 2u || weight->shape[0] != out_features ||
      weight->shape[1] != in_features || !checked_mul(out_features, in_features, weight_elements) ||
      !is_matmul_weight(weight) || !exact_storage(weight, weight_elements, element_bytes) ||
      !vector_f32(bias, out_features))
    return false;
  result = {.weight = weight_view(weight),
            .bias = bias->as_f32(),
            .in_features = in_features,
            .out_features = out_features};
  return true;
}

bool DiffusionHead::load_norm(std::span<const TensorView> weights, std::string_view prefix,
                              std::size_t channels, NormWeights& result) noexcept
{
  std::array<char, 48> weight_key{};
  std::array<char, 48> bias_key{};
  const TensorView* const weight = find_tensor(weights, suffix_key(weight_key, prefix, ".w"));
  const TensorView* const bias = find_tensor(weights, suffix_key(bias_key, prefix, ".b"));
  if (!vector_f32(weight, channels) || !vector_f32(bias, channels))
    return false;
  result = {.weight = weight->as_f32(), .bias = bias->as_f32()};
  return true;
}

bool DiffusionHead::load_residual(std::span<const TensorView> weights, std::string_view prefix,
                                  std::size_t in_channels, std::size_t out_channels,
                                  ResidualWeights& result) noexcept
{
  std::array<char, 48> key_buffer{};
  if (!load_conv(weights, suffix_key(key_buffer, prefix, ".c1"), out_channels, in_channels,
                 cfg_.kernel, result.conv1) ||
      !load_norm(weights, suffix_key(key_buffer, prefix, ".n1"), out_channels, result.norm1) ||
      !load_linear(weights, suffix_key(key_buffer, prefix, ".film"), 2uz * out_channels,
                   cfg_.timestep_dim + cfg_.condition_dim, result.film) ||
      !load_conv(weights, suffix_key(key_buffer, prefix, ".c2"), out_channels, out_channels,
                 cfg_.kernel, result.conv2) ||
      !load_norm(weights, suffix_key(key_buffer, prefix, ".n2"), out_channels, result.norm2))
    return false;
  result.identity_residual = in_channels == out_channels;
  return result.identity_residual || load_conv(weights, suffix_key(key_buffer, prefix, ".res"),
                                               out_channels, in_channels, 1uz, result.residual);
}

bool DiffusionHead::residual_forward(std::span<const float> input, std::size_t length,
                                     const ResidualWeights& weights,
                                     std::span<const float> condition_mish, Arena& arena,
                                     std::span<float> output) noexcept
{
  const std::size_t in_channels = weights.conv1.in_channels;
  const std::size_t out_channels = weights.conv1.out_channels;
  if (input.size() < in_channels * length || output.size() < out_channels * length)
    return false;
  std::byte* const mark = arena.mark();
  std::span<float> first = arena.alloc_span<float, kSimdAlign>(out_channels * length);
  std::span<float> modulation = arena.alloc_span<float, kSimdAlign>(2uz * out_channels);
  if (first.empty() || modulation.empty()) {
    arena.reset_to(mark);
    return false;
  }
  conv1d(input, {weights.conv1.weight, out_channels * in_channels * weights.conv1.kernel},
         {weights.conv1.bias, out_channels}, first, in_channels, out_channels, length, length,
         weights.conv1.kernel, 1uz, weights.conv1.kernel / 2uz, pool_);
  group_norm(first, {weights.norm1.weight, out_channels}, {weights.norm1.bias, out_channels},
             out_channels, length, cfg_.groups);
  mish(first);
  matmul_weight(condition_mish, weights.film.weight, modulation, 1uz, weights.film.in_features,
                weights.film.out_features, pool_);
  add_bias(modulation, {weights.film.bias, weights.film.out_features}, 1uz,
           weights.film.out_features);
  film(first, modulation.first(out_channels), modulation.subspan(out_channels, out_channels),
       out_channels, length);
  conv1d(first, {weights.conv2.weight, out_channels * out_channels * weights.conv2.kernel},
         {weights.conv2.bias, out_channels}, output, out_channels, out_channels, length, length,
         weights.conv2.kernel, 1uz, weights.conv2.kernel / 2uz, pool_);
  group_norm(output, {weights.norm2.weight, out_channels}, {weights.norm2.bias, out_channels},
             out_channels, length, cfg_.groups);
  mish(output);
  if (weights.identity_residual) {
    for (std::size_t i{0uz}; i < output.size(); ++i)
      output[i] += input[i];
  } else {
    conv1d(input, {weights.residual.weight, out_channels * in_channels * weights.residual.kernel},
           {weights.residual.bias, out_channels}, first, in_channels, out_channels, length, length,
           1uz, 1uz, 0uz, pool_);
    for (std::size_t i{0uz}; i < output.size(); ++i)
      output[i] += first[i];
  }
  arena.reset_to(mark);
  return true;
}

bool DiffusionHead::denoise(std::span<const float> condition,
                            std::span<const float> normalized_sample, float timestep,
                            std::span<float> workspace, std::span<float> predicted_noise) noexcept
{
  const std::size_t forward_workspace = workspace_floats_ - (2uz * sample_values());
  if (!ok_ || workspace.size() < forward_workspace)
    return false;
  Arena arena{std::as_writable_bytes(workspace)};
  return denoise_with_arena(condition, normalized_sample, timestep, arena, predicted_noise);
}

bool DiffusionHead::denoise_with_arena(std::span<const float> condition,
                                       std::span<const float> normalized_sample, float timestep,
                                       Arena& arena, std::span<float> predicted_noise) noexcept
{
  const std::size_t values = sample_values();
  if (!ok_ || condition.size() != cfg_.condition_dim || normalized_sample.size() != values ||
      predicted_noise.size() < values || !std::isfinite(timestep))
    return false;

  const std::size_t timestep_hidden = cfg_.timestep_dim * 4uz;
  const std::size_t combined_features = cfg_.timestep_dim + cfg_.condition_dim;
  std::span<float> raw_time = arena.alloc_span<float, kSimdAlign>(cfg_.timestep_dim);
  std::span<float> hidden_time = arena.alloc_span<float, kSimdAlign>(timestep_hidden);
  std::span<float> condition_mish = arena.alloc_span<float, kSimdAlign>(combined_features);
  std::span<float> x = arena.alloc_span<float, kSimdAlign>(values);
  if (raw_time.empty() || hidden_time.empty() || condition_mish.empty() || x.empty())
    return false;
  diffusion_timestep_embedding(timestep, raw_time);
  matmul_weight(raw_time, timestep_in_.weight, hidden_time, 1uz, cfg_.timestep_dim, timestep_hidden,
                pool_);
  add_bias(hidden_time, {timestep_in_.bias, timestep_hidden}, 1uz, timestep_hidden);
  mish(hidden_time);
  matmul_weight(hidden_time, timestep_out_.weight, raw_time, 1uz, timestep_hidden,
                cfg_.timestep_dim, pool_);
  add_bias(raw_time, {timestep_out_.bias, cfg_.timestep_dim}, 1uz, cfg_.timestep_dim);
  std::copy(raw_time.begin(), raw_time.end(), condition_mish.begin());
  std::copy(condition.begin(), condition.end(), condition_mish.begin() + cfg_.timestep_dim);
  mish(condition_mish);
  for (std::size_t t{0uz}; t < cfg_.horizon; ++t)
    for (std::size_t channel{0uz}; channel < cfg_.action_dim; ++channel)
      x[(channel * cfg_.horizon) + t] = normalized_sample[(t * cfg_.action_dim) + channel];

  std::array<std::span<float>, kMaxStages> skips{};
  std::array<std::size_t, kMaxStages> skip_lengths{};
  std::size_t length = cfg_.horizon;
  std::size_t channels = cfg_.action_dim;
  for (std::size_t stage{0uz}; stage < cfg_.stages; ++stage) {
    for (std::size_t block{0uz}; block < 2uz; ++block) {
      const std::size_t out_channels = dims_[stage];
      std::span<float> output = arena.alloc_span<float, kSimdAlign>(out_channels * length);
      if (output.empty() || !residual_forward(x, length, down_[stage].residuals[block],
                                              condition_mish, arena, output))
        return false;
      x = output;
      channels = out_channels;
    }
    skips[stage] = x;
    skip_lengths[stage] = length;
    if (!down_[stage].identity_downsample) {
      const std::size_t output_length = length / 2uz;
      std::span<float> output = arena.alloc_span<float, kSimdAlign>(channels * output_length);
      if (output.empty())
        return false;
      const ConvWeights& weights = down_[stage].downsample;
      conv1d(x, {weights.weight, channels * channels * weights.kernel}, {weights.bias, channels},
             output, channels, channels, length, output_length, weights.kernel, 2uz, 1uz, pool_);
      x = output;
      length = output_length;
    }
  }

  for (const ResidualWeights& weights : middle_) {
    std::span<float> output = arena.alloc_span<float, kSimdAlign>(channels * length);
    if (output.empty() || !residual_forward(x, length, weights, condition_mish, arena, output))
      return false;
    x = output;
  }

  for (std::size_t stage{0uz}; stage + 1uz < cfg_.stages; ++stage) {
    const std::size_t skip_index = cfg_.stages - 1uz - stage;
    const std::span<float> skip = skips[skip_index];
    if (skip_lengths[skip_index] != length || skip.size() != x.size())
      return false;
    std::span<float> concatenated = arena.alloc_span<float, kSimdAlign>(x.size() + skip.size());
    if (concatenated.empty())
      return false;
    std::copy(x.begin(), x.end(), concatenated.begin());
    std::copy(skip.begin(), skip.end(), concatenated.begin() + x.size());
    x = concatenated;
    channels *= 2uz;
    for (std::size_t block{0uz}; block < 2uz; ++block) {
      const std::size_t out_channels = dims_[cfg_.stages - 2uz - stage];
      std::span<float> output = arena.alloc_span<float, kSimdAlign>(out_channels * length);
      if (output.empty() ||
          !residual_forward(x, length, up_[stage].residuals[block], condition_mish, arena, output))
        return false;
      x = output;
      channels = out_channels;
    }
    const std::size_t output_length = length * 2uz;
    std::span<float> output = arena.alloc_span<float, kSimdAlign>(channels * output_length);
    if (output.empty())
      return false;
    const ConvWeights& weights = up_[stage].upsample;
    conv_transpose1d(x, {weights.weight, channels * channels * weights.kernel},
                     {weights.bias, channels}, output, channels, channels, length, output_length,
                     weights.kernel, 2uz, 1uz, pool_);
    x = output;
    length = output_length;
  }

  if (channels != dims_[0] || length != cfg_.horizon)
    return false;
  std::span<float> final = arena.alloc_span<float, kSimdAlign>(channels * length);
  std::span<float> result = arena.alloc_span<float, kSimdAlign>(values);
  if (final.empty() || result.empty())
    return false;
  conv1d(x, {final_conv_.weight, channels * channels * final_conv_.kernel},
         {final_conv_.bias, channels}, final, channels, channels, length, length,
         final_conv_.kernel, 1uz, final_conv_.kernel / 2uz, pool_);
  group_norm(final, {final_norm_.weight, channels}, {final_norm_.bias, channels}, channels, length,
             kFinalNormGroups);
  mish(final);
  conv1d(final, {output_conv_.weight, cfg_.action_dim * channels},
         {output_conv_.bias, cfg_.action_dim}, result, channels, cfg_.action_dim, length, length,
         1uz, 1uz, 0uz, pool_);
  for (std::size_t t{0uz}; t < cfg_.horizon; ++t)
    for (std::size_t channel{0uz}; channel < cfg_.action_dim; ++channel)
      predicted_noise[(t * cfg_.action_dim) + channel] = result[(channel * length) + t];
  return true;
}

bool DiffusionHead::sample(std::span<const float> condition, std::span<const float> initial_noise,
                           std::size_t inference_steps, Scheduler scheduler, std::uint64_t seed,
                           std::span<float> workspace, std::span<float> action) noexcept
{
  const std::size_t values = sample_values();
  if (!ok_ || condition.size() != cfg_.condition_dim || initial_noise.size() != values ||
      action.size() < values || workspace.size() < workspace_floats_ || inference_steps == 0uz ||
      inference_steps > cfg_.train_timesteps || (scheduler != kDDIM && scheduler != kDDPM))
    return false;
  std::span<float> x = workspace.first(values);
  std::span<float> predicted_noise = workspace.subspan(values, values);
  std::span<float> denoiser_workspace = workspace.subspan(2uz * values);
  std::copy(initial_noise.begin(), initial_noise.end(), x.begin());
  GaussianGenerator gaussian{seed};
  const std::size_t step_ratio = cfg_.train_timesteps / inference_steps;
  for (std::size_t index{0uz}; index < inference_steps; ++index) {
    const std::size_t reverse_index = inference_steps - 1uz - index;
    const std::size_t timestep = reverse_index * step_ratio;
    const std::ptrdiff_t previous =
        static_cast<std::ptrdiff_t>(timestep) - static_cast<std::ptrdiff_t>(step_ratio);
    if (!denoise(condition, x, static_cast<float>(timestep), denoiser_workspace, predicted_noise))
      return false;
    const float alpha_t = alphas_cumprod_[timestep];
    const float alpha_previous = previous >= 0 ? alphas_cumprod_[previous] : 1.0F;
    const float beta_t = 1.0F - alpha_t;
    if (scheduler == kDDIM) {
      for (std::size_t value{0uz}; value < values; ++value) {
        float predicted_original =
            (x[value] - (std::sqrt(beta_t) * predicted_noise[value])) / std::sqrt(alpha_t);
        if (cfg_.clip_sample)
          predicted_original =
              std::clamp(predicted_original, -cfg_.clip_sample_range, cfg_.clip_sample_range);
        x[value] = (std::sqrt(alpha_previous) * predicted_original) +
                   (std::sqrt(1.0F - alpha_previous) * predicted_noise[value]);
      }
    } else {
      const float current_alpha = alpha_t / alpha_previous;
      const float current_beta = 1.0F - current_alpha;
      const float original_coefficient =
          std::sqrt(alpha_previous) * current_beta / std::max(beta_t, 1e-20F);
      const float sample_coefficient =
          std::sqrt(current_alpha) * (1.0F - alpha_previous) / std::max(beta_t, 1e-20F);
      const float variance = ((1.0F - alpha_previous) / std::max(beta_t, 1e-20F)) * current_beta;
      for (std::size_t value{0uz}; value < values; ++value) {
        float predicted_original =
            (x[value] - (std::sqrt(beta_t) * predicted_noise[value])) / std::sqrt(alpha_t);
        if (cfg_.clip_sample)
          predicted_original =
              std::clamp(predicted_original, -cfg_.clip_sample_range, cfg_.clip_sample_range);
        float next = (original_coefficient * predicted_original) + (sample_coefficient * x[value]);
        if (previous >= 0)
          next += std::sqrt(std::max(variance, 1e-20F)) * gaussian.next();
        x[value] = next;
      }
    }
  }

  for (std::size_t t{0uz}; t < cfg_.horizon; ++t) {
    for (std::size_t channel{0uz}; channel < cfg_.action_dim; ++channel) {
      const std::size_t index = (t * cfg_.action_dim) + channel;
      action[index] = ((x[index] + 1.0F) * 0.5F * (action_max_[channel] - action_min_[channel])) +
                      action_min_[channel];
    }
  }
  return true;
}

} // namespace fe
