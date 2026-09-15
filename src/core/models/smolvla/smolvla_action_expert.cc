#include "models/smolvla/smolvla_action_expert.h"

#include "kernels/kernels.h"
#include "kernels/span_ops.h"
#include "loader/weight_ops.h"

#include <algorithm>
#include <bit>
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

[[nodiscard]] float bf16_to_float(std::uint16_t value) noexcept
{
  return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16u);
}

[[nodiscard]] bool copy_weight_vector(const TensorView* source,
                                      std::span<float> destination) noexcept
{
  if (!vector_weight(source, destination.size()))
    return false;
  if (source->is_f32()) {
    std::copy_n(source->as_f32(), destination.size(), destination.data());
    return true;
  }
  const std::uint16_t* const values = source->as_bf16();
  for (std::size_t i{}; i < destination.size(); ++i)
    destination[i] = bf16_to_float(values[i]);
  return true;
}

void round_to_bf16(std::span<float> values) noexcept
{
  for (float& value : values) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t bias = 0x7FFFu + ((bits >> 16u) & 1u);
    value = std::bit_cast<float>((bits + bias) & 0xFFFF0000u);
  }
}

// The upstream SmolLM expert normalizes in float32 with rms_norm_eps=1e-5,
// then casts the result to BF16 for its projections.  The generic kernel has
// a different fixed epsilon, so keep this small, allocation-free operation
// local to the checkpoint-specific execution path.
void smolvla_rmsnorm(std::span<const float> input, std::span<const float> weight,
                     std::span<float> output, std::size_t rows, std::size_t width) noexcept
{
  constexpr float kEpsilon = 1e-5F;
  for (std::size_t row{}; row < rows; ++row) {
    const float* const source = input.data() + (row * width);
    float* const destination = output.data() + (row * width);
    float squared_sum{};
    for (std::size_t channel{}; channel < width; ++channel)
      squared_sum += source[channel] * source[channel];
    const float scale = 1.0F / std::sqrt((squared_sum / static_cast<float>(width)) + kEpsilon);
    for (std::size_t channel{}; channel < width; ++channel)
      destination[channel] = source[channel] * scale * weight[channel];
  }
}

void apply_rope(std::span<float> values, std::size_t rows, std::size_t heads,
                std::size_t head_width, std::size_t position_base) noexcept
{
  constexpr float kLogWavelength = 9.210340371976184F; // log(10,000)
  const std::size_t half = head_width / 2uz;
  for (std::size_t row{}; row < rows; ++row) {
    const float position = static_cast<float>(position_base + row);
    for (std::size_t head{}; head < heads; ++head) {
      float* const data = values.data() + ((row * heads + head) * head_width);
      for (std::size_t channel{}; channel < half; ++channel) {
        const float exponent =
            (2.0F * static_cast<float>(channel)) / static_cast<float>(head_width);
        const float angle = position / std::exp(kLogWavelength * exponent);
        const float sine = std::sin(angle);
        const float cosine = std::cos(angle);
        const float first = data[channel];
        const float second = data[half + channel];
        data[channel] = (first * cosine) - (second * sine);
        data[half + channel] = (second * cosine) + (first * sine);
      }
    }
  }
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

  const TensorView* const final_norm =
      find_tensor(weights, "model.vlm_with_expert.lm_expert.norm.weight");
  const TensorView* const first_q = layer_tensor(weights, 0uz, "self_attn.q_proj.weight");
  const TensorView* const first_k = layer_tensor(weights, 0uz, "self_attn.k_proj.weight");
  const TensorView* const first_gate = layer_tensor(weights, 0uz, "mlp.gate_proj.weight");
  if (first_q == nullptr || first_k == nullptr || first_gate == nullptr || first_q->ndim != 2uz ||
      first_k->ndim != 2uz || first_gate->ndim != 2uz)
    return;

  cfg_.attention_width = first_q->shape[0];
  cfg_.key_value_width = first_k->shape[0];
  cfg_.mlp_width = first_gate->shape[0];
  // SmolVLA's Gemma expert has 12 query and 4 KV heads of width 80. These
  // dimensions are encoded by the pinned checkpoint and kept explicit rather
  // than guessing an ambiguous head decomposition from matrix shapes.
  constexpr std::size_t kHeadWidth = 80uz;
  if (cfg_.attention_width != 960uz || cfg_.key_value_width != 320uz || cfg_.mlp_width != 2048uz ||
      (cfg_.attention_width % kHeadWidth) != 0uz || (cfg_.key_value_width % kHeadWidth) != 0uz ||
      (cfg_.attention_width / cfg_.key_value_width) != 3uz ||
      !vector_weight(final_norm, cfg_.expert_width))
    return;

  std::array<Layer, kMaxExpertLayers> loaded_layers{};
  for (std::size_t layer{}; layer < kMaxExpertLayers; ++layer) {
    const TensorView* const input_norm = layer_tensor(weights, layer, "input_layernorm.weight");
    const TensorView* const post_norm =
        layer_tensor(weights, layer, "post_attention_layernorm.weight");
    const TensorView* const q = layer_tensor(weights, layer, "self_attn.q_proj.weight");
    const TensorView* const k = layer_tensor(weights, layer, "self_attn.k_proj.weight");
    const TensorView* const v = layer_tensor(weights, layer, "self_attn.v_proj.weight");
    const TensorView* const o = layer_tensor(weights, layer, "self_attn.o_proj.weight");
    const TensorView* const gate = layer_tensor(weights, layer, "mlp.gate_proj.weight");
    const TensorView* const up = layer_tensor(weights, layer, "mlp.up_proj.weight");
    const TensorView* const down = layer_tensor(weights, layer, "mlp.down_proj.weight");
    if (input_norm == nullptr && post_norm == nullptr && q == nullptr && k == nullptr &&
        v == nullptr && o == nullptr && gate == nullptr && up == nullptr && down == nullptr)
      break;
    const bool cross_attention = (layer % 2uz) != 0uz;
    const std::size_t kv_input = cross_attention ? cfg_.key_value_width : cfg_.expert_width;
    if (!vector_weight(input_norm, cfg_.expert_width) ||
        !vector_weight(post_norm, cfg_.expert_width) ||
        !matrix_weight(q, cfg_.attention_width, cfg_.expert_width) ||
        !matrix_weight(k, cfg_.key_value_width, kv_input) ||
        !matrix_weight(v, cfg_.key_value_width, kv_input) ||
        !matrix_weight(o, cfg_.expert_width, cfg_.attention_width) ||
        !matrix_weight(gate, cfg_.mlp_width, cfg_.expert_width) ||
        !matrix_weight(up, cfg_.mlp_width, cfg_.expert_width) ||
        !matrix_weight(down, cfg_.expert_width, cfg_.mlp_width))
      return;
    loaded_layers[cfg_.expert_layers] = {
        .q_proj = weight(q),
        .k_proj = weight(k),
        .v_proj = weight(v),
        .o_proj = weight(o),
        .gate_proj = weight(gate),
        .up_proj = weight(up),
        .down_proj = weight(down),
        .cross_attention = cross_attention,
    };
    ++cfg_.expert_layers;
  }
  if (cfg_.expert_layers == 0uz)
    return;

  std::byte* const persistent_mark = arena_->mark();
  for (std::size_t layer{}; layer < cfg_.expert_layers; ++layer) {
    const TensorView* const input_norm = layer_tensor(weights, layer, "input_layernorm.weight");
    const TensorView* const post_norm =
        layer_tensor(weights, layer, "post_attention_layernorm.weight");
    float* const input = arena_->alloc_array<float, kSimdAlign>(cfg_.expert_width);
    float* const post = arena_->alloc_array<float, kSimdAlign>(cfg_.expert_width);
    if (input == nullptr || post == nullptr ||
        !copy_weight_vector(input_norm, {input, cfg_.expert_width}) ||
        !copy_weight_vector(post_norm, {post, cfg_.expert_width})) {
      arena_->reset_to(persistent_mark);
      return;
    }
    loaded_layers[layer].input_norm = input;
    loaded_layers[layer].post_attention_norm = post;
  }
  float* const final = arena_->alloc_array<float, kSimdAlign>(cfg_.expert_width);
  if (final == nullptr || !copy_weight_vector(final_norm, {final, cfg_.expert_width})) {
    arena_->reset_to(persistent_mark);
    return;
  }

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
  layers_ = loaded_layers;
  final_norm_ = final;
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

bool SmolVLAActionExpert::run_with_prefix_kv(std::span<const float> suffix,
                                             std::span<const float> prefix_keys,
                                             std::span<const float> prefix_values,
                                             std::span<const std::uint8_t> prefix_mask,
                                             std::span<float> output) noexcept
{
  if (!valid_ || suffix.empty() || suffix.size() % cfg_.expert_width != 0uz ||
      output.size() != suffix.size() || prefix_mask.empty() || prefix_mask.size() > 512uz)
    return false;
  const std::size_t chunk_size = suffix.size() / cfg_.expert_width;
  const std::size_t prefix_size = prefix_mask.size();
  if (chunk_size > 512uz ||
      prefix_size > (std::numeric_limits<std::size_t>::max() / cfg_.expert_layers) ||
      prefix_size * cfg_.expert_layers >
          (std::numeric_limits<std::size_t>::max() / cfg_.key_value_width))
    return false;
  const std::size_t cache_values = cfg_.expert_layers * prefix_size * cfg_.key_value_width;
  if (prefix_keys.size() != cache_values || prefix_values.size() != cache_values)
    return false;
  std::size_t valid_prefix{};
  bool padding_started{};
  for (const std::uint8_t value : prefix_mask) {
    if (value > 1u)
      return false;
    if (value == 0u) {
      padding_started = true;
      continue;
    }
    if (padding_started)
      return false;
    ++valid_prefix;
  }
  if (valid_prefix == 0uz)
    return false;

  copy_span(suffix, output);
  constexpr std::size_t kHeadWidth = 80uz;
  const std::size_t query_heads = cfg_.attention_width / kHeadWidth;
  const std::size_t key_value_heads = cfg_.key_value_width / kHeadWidth;
  const std::size_t groups = query_heads / key_value_heads;
  for (std::size_t layer_index{}; layer_index < cfg_.expert_layers; ++layer_index) {
    const Layer& layer = layers_[layer_index];
    std::byte* const mark = arena_->mark();
    const std::span<float> normed = scratch(chunk_size * cfg_.expert_width);
    const std::span<float> query = scratch(chunk_size * cfg_.attention_width);
    const std::size_t key_rows = layer.cross_attention ? prefix_size : chunk_size;
    const std::span<float> keys = scratch(key_rows * cfg_.key_value_width);
    const std::span<float> values = scratch(key_rows * cfg_.key_value_width);
    const std::span<float> attended = scratch(chunk_size * cfg_.attention_width);
    const std::span<float> projected = scratch(chunk_size * cfg_.expert_width);
    const std::span<float> gate = scratch(chunk_size * cfg_.mlp_width);
    const std::span<float> up = scratch(chunk_size * cfg_.mlp_width);
    const std::span<float> scores = scratch(prefix_size + chunk_size);
    if (normed.empty() || query.empty() || keys.empty() || values.empty() || attended.empty() ||
        projected.empty() || gate.empty() || up.empty() || scores.empty()) {
      arena_->reset_to(mark);
      return false;
    }

    smolvla_rmsnorm(output, {layer.input_norm, cfg_.expert_width}, normed, chunk_size,
                    cfg_.expert_width);
    round_to_bf16(normed);
    matmul_weight(normed, layer.q_proj, query, chunk_size, cfg_.expert_width, cfg_.attention_width,
                  pool_);
    round_to_bf16(query);
    if (layer.cross_attention) {
      const std::size_t offset = layer_index * prefix_size * cfg_.key_value_width;
      matmul_weight(prefix_keys.subspan(offset, prefix_size * cfg_.key_value_width), layer.k_proj,
                    keys, prefix_size, cfg_.key_value_width, cfg_.key_value_width, pool_);
      matmul_weight(prefix_values.subspan(offset, prefix_size * cfg_.key_value_width), layer.v_proj,
                    values, prefix_size, cfg_.key_value_width, cfg_.key_value_width, pool_);
      apply_rope(query, chunk_size, query_heads, kHeadWidth, 0uz);
    } else {
      matmul_weight(normed, layer.k_proj, keys, chunk_size, cfg_.expert_width, cfg_.key_value_width,
                    pool_);
      matmul_weight(normed, layer.v_proj, values, chunk_size, cfg_.expert_width,
                    cfg_.key_value_width, pool_);
      round_to_bf16(keys);
      round_to_bf16(values);
      apply_rope(query, chunk_size, query_heads, kHeadWidth, valid_prefix);
      apply_rope(keys, chunk_size, key_value_heads, kHeadWidth, valid_prefix);
    }
    round_to_bf16(query);
    if (!layer.cross_attention)
      round_to_bf16(keys);

    for (std::size_t row{}; row < chunk_size; ++row) {
      for (std::size_t head{}; head < query_heads; ++head) {
        const std::size_t kv_head = head / groups;
        const float* const q = query.data() + ((row * query_heads + head) * kHeadWidth);
        const std::size_t attended_offset = (row * cfg_.attention_width) + (head * kHeadWidth);
        const std::size_t score_count =
            layer.cross_attention ? prefix_size : prefix_size + row + 1uz;
        for (std::size_t token{}; token < prefix_size; ++token) {
          if (prefix_mask[token] == 0u) {
            scores[token] = -std::numeric_limits<float>::infinity();
            continue;
          }
          const float* const key =
              layer.cross_attention
                  ? keys.data() + ((token * key_value_heads + kv_head) * kHeadWidth)
                  : prefix_keys.data() +
                        (((layer_index * prefix_size + token) * key_value_heads + kv_head) *
                         kHeadWidth);
          float dot{};
          for (std::size_t channel{}; channel < kHeadWidth; ++channel)
            dot += q[channel] * key[channel];
          scores[token] = dot / std::sqrt(static_cast<float>(kHeadWidth));
        }
        if (!layer.cross_attention) {
          for (std::size_t token{}; token <= row; ++token) {
            const float* const key =
                keys.data() + ((token * key_value_heads + kv_head) * kHeadWidth);
            float dot{};
            for (std::size_t channel{}; channel < kHeadWidth; ++channel)
              dot += q[channel] * key[channel];
            scores[prefix_size + token] = dot / std::sqrt(static_cast<float>(kHeadWidth));
          }
        }
        std::span<float> probabilities = scores.first(score_count);
        softmax(probabilities);
        // Cross-attention K/V projections are F32 in the pinned checkpoint,
        // so the upstream implementation keeps its attention probabilities
        // and values in F32. Self-attention consumes BF16 VLM/expert values.
        if (!layer.cross_attention)
          round_to_bf16(probabilities);
        for (std::size_t channel{}; channel < kHeadWidth; ++channel) {
          float sum{};
          for (std::size_t token{}; token < prefix_size; ++token) {
            if (prefix_mask[token] == 0u)
              continue;
            const float* const value =
                layer.cross_attention
                    ? values.data() + ((token * key_value_heads + kv_head) * kHeadWidth)
                    : prefix_values.data() +
                          (((layer_index * prefix_size + token) * key_value_heads + kv_head) *
                           kHeadWidth);
            sum += probabilities[token] * value[channel];
          }
          if (!layer.cross_attention)
            for (std::size_t token{}; token <= row; ++token)
              sum += probabilities[prefix_size + token] *
                     values[(token * cfg_.key_value_width) + (kv_head * kHeadWidth) + channel];
          attended[attended_offset + channel] = sum;
        }
      }
    }
    round_to_bf16(attended);
    matmul_weight(attended, layer.o_proj, projected, chunk_size, cfg_.attention_width,
                  cfg_.expert_width, pool_);
    round_to_bf16(projected);
    for (std::size_t i{}; i < output.size(); ++i)
      output[i] += projected[i];
    round_to_bf16(output);

    smolvla_rmsnorm(output, {layer.post_attention_norm, cfg_.expert_width}, normed, chunk_size,
                    cfg_.expert_width);
    round_to_bf16(normed);
    matmul_weight(normed, layer.gate_proj, gate, chunk_size, cfg_.expert_width, cfg_.mlp_width,
                  pool_);
    matmul_weight(normed, layer.up_proj, up, chunk_size, cfg_.expert_width, cfg_.mlp_width, pool_);
    round_to_bf16(gate);
    round_to_bf16(up);
    gate_silu(up, gate, gate);
    round_to_bf16(gate);
    matmul_weight(gate, layer.down_proj, projected, chunk_size, cfg_.mlp_width, cfg_.expert_width,
                  pool_);
    round_to_bf16(projected);
    for (std::size_t i{}; i < output.size(); ++i)
      output[i] += projected[i];
    round_to_bf16(output);
    arena_->reset_to(mark);
  }
  std::byte* const mark = arena_->mark();
  const std::span<float> normalized = scratch(output.size());
  if (normalized.empty()) {
    arena_->reset_to(mark);
    return false;
  }
  smolvla_rmsnorm(output, {final_norm_, cfg_.expert_width}, normalized, chunk_size,
                  cfg_.expert_width);
  round_to_bf16(normalized);
  copy_span(normalized, output);
  arena_->reset_to(mark);
  return true;
}

bool SmolVLAActionExpert::denoise_with_prefix_kv(std::span<const float> noisy_actions,
                                                 float timestep, std::span<const float> prefix_keys,
                                                 std::span<const float> prefix_values,
                                                 std::span<const std::uint8_t> prefix_mask,
                                                 std::span<float> output) noexcept
{
  if (!valid_ || !std::isfinite(timestep) || noisy_actions.empty() ||
      noisy_actions.size() % cfg_.max_action_dim != 0uz)
    return false;
  const std::size_t chunk_size = noisy_actions.size() / cfg_.max_action_dim;
  if (output.size() != chunk_size * cfg_.max_action_dim)
    return false;
  std::byte* const mark = arena_->mark();
  const std::span<float> suffix = scratch(chunk_size * cfg_.expert_width);
  const std::span<float> hidden = scratch(chunk_size * cfg_.expert_width);
  if (suffix.empty() || hidden.empty() || !embed_suffix(noisy_actions, timestep, suffix) ||
      !run_with_prefix_kv(suffix, prefix_keys, prefix_values, prefix_mask, hidden) ||
      !project_actions(hidden, output)) {
    arena_->reset_to(mark);
    return false;
  }
  arena_->reset_to(mark);
  return true;
}

} // namespace fe
