#include "models/smolvla/smolvla_action_expert.h"

#include "kernels/kernels.h"
#include "loader/weight_ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

namespace fe {
namespace {

[[nodiscard]] bool vector_f32(const TensorView* tensor, std::size_t size) noexcept
{
  return tensor != nullptr && tensor->is_f32() && tensor->ndim == 1uz && tensor->shape[0] == size;
}

[[nodiscard]] bool vector_weight(const TensorView* tensor, std::size_t size) noexcept
{
  return tensor != nullptr && tensor->ndim == 1uz && tensor->shape[0] == size &&
         (tensor->is_f32() || tensor->is_bf16());
}

[[nodiscard]] bool matrix_weight(const TensorView* tensor, std::size_t rows,
                                 std::size_t columns) noexcept
{
  return tensor != nullptr && tensor->ndim == 2uz && tensor->shape[0] == rows &&
         tensor->shape[1] == columns && (tensor->is_f32() || tensor->is_bf16());
}

[[nodiscard]] WeightView weight(const TensorView* tensor) noexcept
{
  return tensor == nullptr ? WeightView{} : WeightView{tensor->data, tensor->dtype};
}

[[nodiscard]] const TensorView* layer_tensor(std::span<const TensorView> tensors, std::size_t layer,
                                             std::string_view suffix) noexcept
{
  std::string name = "model.vlm_with_expert.lm_expert.layers." + std::to_string(layer) + ".";
  name.append(suffix);
  return find_tensor(tensors, name);
}

[[nodiscard]] bool layer_present(std::span<const TensorView> tensors, std::size_t layer,
                                 std::size_t expert_width, std::size_t vlm_width) noexcept
{
  const TensorView* const norm = layer_tensor(tensors, layer, "input_layernorm.weight");
  const TensorView* const q = layer_tensor(tensors, layer, "self_attn.q_proj.weight");
  const TensorView* const mlp = layer_tensor(tensors, layer, "mlp.gate_proj.weight");
  return vector_weight(norm, expert_width) && matrix_weight(q, vlm_width, expert_width) &&
         mlp != nullptr && mlp->ndim == 2uz && mlp->shape[1] == expert_width &&
         mlp->shape[0] > expert_width && (mlp->is_f32() || mlp->is_bf16());
}

} // namespace

SmolVLAActionExpert::SmolVLAActionExpert(std::span<const TensorView> weights, Arena& arena) noexcept
    : arena_{&arena}
{
  const TensorView* const action_in = find_tensor(weights, "model.action_in_proj.weight");
  const TensorView* const action_out = find_tensor(weights, "model.action_out_proj.weight");
  const TensorView* const time_in = find_tensor(weights, "model.action_time_mlp_in.weight");
  const TensorView* const time_out = find_tensor(weights, "model.action_time_mlp_out.weight");
  const TensorView* const state = find_tensor(weights, "model.state_proj.weight");
  if (action_in == nullptr || action_out == nullptr || time_in == nullptr || time_out == nullptr ||
      state == nullptr || action_in->ndim != 2uz || action_out->ndim != 2uz ||
      time_in->ndim != 2uz || time_out->ndim != 2uz || state->ndim != 2uz)
    return;

  cfg_.expert_width = action_in->shape[0];
  cfg_.max_action_dim = action_in->shape[1];
  cfg_.vlm_width = state->shape[0];
  cfg_.max_state_dim = state->shape[1];
  if (cfg_.expert_width == 0uz || cfg_.expert_width > 4096uz || cfg_.max_action_dim == 0uz ||
      cfg_.max_action_dim > 256uz || cfg_.vlm_width == 0uz || cfg_.vlm_width > 4096uz ||
      cfg_.max_state_dim == 0uz || cfg_.max_state_dim > 256uz || (cfg_.expert_width % 2uz) != 0uz ||
      !matrix_weight(action_in, cfg_.expert_width, cfg_.max_action_dim) ||
      !matrix_weight(action_out, cfg_.max_action_dim, cfg_.expert_width) ||
      !matrix_weight(time_in, cfg_.expert_width, 2uz * cfg_.expert_width) ||
      !matrix_weight(time_out, cfg_.expert_width, cfg_.expert_width) ||
      !matrix_weight(state, cfg_.vlm_width, cfg_.max_state_dim))
    return;

  const TensorView* const action_in_bias = find_tensor(weights, "model.action_in_proj.bias");
  const TensorView* const action_out_bias = find_tensor(weights, "model.action_out_proj.bias");
  const TensorView* const time_in_bias = find_tensor(weights, "model.action_time_mlp_in.bias");
  const TensorView* const time_out_bias = find_tensor(weights, "model.action_time_mlp_out.bias");
  const TensorView* const state_bias = find_tensor(weights, "model.state_proj.bias");
  if (!vector_f32(action_in_bias, cfg_.expert_width) ||
      !vector_f32(action_out_bias, cfg_.max_action_dim) ||
      !vector_f32(time_in_bias, cfg_.expert_width) ||
      !vector_f32(time_out_bias, cfg_.expert_width) || !vector_f32(state_bias, cfg_.vlm_width))
    return;

  for (std::size_t layer{};
       layer < kMaxExpertLayers && layer_present(weights, layer, cfg_.expert_width, cfg_.vlm_width);
       ++layer)
    ++cfg_.expert_layers;
  if (cfg_.expert_layers == 0uz)
    return;

  state_proj_ = weight(state);
  state_bias_ = state_bias->as_f32();
  action_in_proj_ = weight(action_in);
  action_in_bias_ = action_in_bias->as_f32();
  action_out_proj_ = weight(action_out);
  action_out_bias_ = action_out_bias->as_f32();
  time_mlp_in_ = weight(time_in);
  time_mlp_in_bias_ = time_in_bias->as_f32();
  time_mlp_out_ = weight(time_out);
  time_mlp_out_bias_ = time_out_bias->as_f32();
  valid_ = true;
}

bool SmolVLAActionExpert::embed_suffix(std::span<const float> noisy_actions, float timestep,
                                       std::span<float> output) noexcept
{
  if (!valid_ || !std::isfinite(timestep) || noisy_actions.empty() ||
      noisy_actions.size() % cfg_.max_action_dim != 0uz ||
      output.size() != (noisy_actions.size() / cfg_.max_action_dim) * cfg_.expert_width)
    return false;
  const std::size_t rows = noisy_actions.size() / cfg_.max_action_dim;
  std::byte* const mark = arena_->mark();
  const std::span<float> action = scratch(rows * cfg_.expert_width);
  const std::span<float> fused = scratch(rows * (2uz * cfg_.expert_width));
  const std::span<float> time = scratch(cfg_.expert_width);
  if (action.empty() || fused.empty() || time.empty()) {
    arena_->reset_to(mark);
    return false;
  }
  matmul_weight(noisy_actions, action_in_proj_, action, rows, cfg_.max_action_dim,
                cfg_.expert_width, pool_);
  for (std::size_t row{}; row < rows; ++row)
    for (std::size_t channel{}; channel < cfg_.expert_width; ++channel)
      action[(row * cfg_.expert_width) + channel] += action_in_bias_[channel];

  const std::size_t half = cfg_.expert_width / 2uz;
  for (std::size_t i{}; i < half; ++i) {
    const float fraction = static_cast<float>(i) / static_cast<float>(half - 1uz);
    const float period = 0.004F * std::pow(4.0F / 0.004F, fraction);
    constexpr float kPi = 3.14159265358979323846F;
    const float angle = (2.0F * kPi / period) * timestep;
    time[i] = std::sin(angle);
    time[half + i] = std::cos(angle);
  }
  for (std::size_t row{}; row < rows; ++row) {
    std::copy_n(action.data() + (row * cfg_.expert_width), cfg_.expert_width,
                fused.data() + (row * 2uz * cfg_.expert_width));
    std::copy_n(time.data(), cfg_.expert_width,
                fused.data() + (row * 2uz * cfg_.expert_width) + cfg_.expert_width);
  }
  matmul_weight(fused, time_mlp_in_, output, rows, 2uz * cfg_.expert_width, cfg_.expert_width,
                pool_);
  for (std::size_t i{}; i < rows * cfg_.expert_width; ++i)
    output[i] = output[i] + time_mlp_in_bias_[i % cfg_.expert_width];
  silu(output);
  matmul_weight(output, time_mlp_out_, fused, rows, cfg_.expert_width, cfg_.expert_width, pool_);
  for (std::size_t i{}; i < rows * cfg_.expert_width; ++i)
    output[i] = fused[i] + time_mlp_out_bias_[i % cfg_.expert_width];
  arena_->reset_to(mark);
  return true;
}

bool SmolVLAActionExpert::project_state(std::span<const float> state,
                                        std::span<float> output) noexcept
{
  if (!valid_ || state.size() != cfg_.max_state_dim || output.size() != cfg_.vlm_width)
    return false;
  matmul_weight(state, state_proj_, output, 1uz, cfg_.max_state_dim, cfg_.vlm_width, pool_);
  for (std::size_t i{}; i < cfg_.vlm_width; ++i)
    output[i] += state_bias_[i];
  return true;
}

bool SmolVLAActionExpert::project_actions(std::span<const float> hidden,
                                          std::span<float> output) noexcept
{
  if (!valid_ || hidden.empty() || hidden.size() % cfg_.expert_width != 0uz ||
      output.size() != (hidden.size() / cfg_.expert_width) * cfg_.max_action_dim)
    return false;
  const std::size_t rows = hidden.size() / cfg_.expert_width;
  matmul_weight(hidden, action_out_proj_, output, rows, cfg_.expert_width, cfg_.max_action_dim,
                pool_);
  for (std::size_t row{}; row < rows; ++row)
    for (std::size_t i{}; i < cfg_.max_action_dim; ++i)
      output[(row * cfg_.max_action_dim) + i] += action_out_bias_[i];
  return true;
}

} // namespace fe
