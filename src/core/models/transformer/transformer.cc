#include "models/transformer/transformer.h"

#include "kernels/kernels.h"
#include "kernels/span_ops.h"
#include "loader/weight_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace fe {
namespace {

[[nodiscard]] bool vector_f32(const TensorView* tensor, std::size_t size) noexcept
{
  return tensor != nullptr && tensor->is_f32() && tensor->ndim == 1uz && tensor->shape[0] == size;
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
  char name[64]{};
  const int written = std::snprintf(name, sizeof(name), "transformer.layers.%zu.%.*s", layer,
                                    static_cast<int>(suffix.size()), suffix.data());
  return written > 0 && static_cast<std::size_t>(written) < sizeof(name)
             ? find_tensor(tensors, name)
             : nullptr;
}

[[nodiscard]] bool config_value(float value, std::size_t& output) noexcept
{
  if (!std::isfinite(value) || value < 1.0F ||
      value > static_cast<float>(std::numeric_limits<std::size_t>::max()))
    return false;
  const auto converted = static_cast<std::size_t>(value);
  if (static_cast<float>(converted) != value)
    return false;
  output = converted;
  return true;
}

} // namespace

Transformer::Transformer(std::span<const TensorView> weights, Arena& arena) noexcept
    : arena_{&arena}
{
  const TensorView* const config = find_tensor(weights, "transformer.config");
  if (config == nullptr || !config->is_f32() || config->ndim != 1uz || config->shape[0] != 6uz)
    return;
  const float* const values = config->as_f32();
  if (!config_value(values[0], cfg_.vocab) || !config_value(values[1], cfg_.d_model) ||
      !config_value(values[2], cfg_.n_layers) || !config_value(values[3], cfg_.n_heads) ||
      !config_value(values[4], cfg_.max_sequence) || !config_value(values[5], cfg_.mlp_dim) ||
      cfg_.vocab > 1'000'000uz || cfg_.d_model > 16'384uz || cfg_.n_layers > kMaxLayers ||
      cfg_.n_heads > 256uz || cfg_.max_sequence > 4'096uz || cfg_.mlp_dim > 65'536uz ||
      cfg_.d_model % cfg_.n_heads != 0uz)
    return;
  const TensorView* const token = find_tensor(weights, "transformer.embeddings.token.weight");
  const TensorView* const position = find_tensor(weights, "transformer.embeddings.position.weight");
  const TensorView* const final_weight = find_tensor(weights, "transformer.norm_f.weight");
  const TensorView* const final_bias = find_tensor(weights, "transformer.norm_f.bias");
  if (!matrix_weight(token, cfg_.vocab, cfg_.d_model) ||
      !matrix_weight(position, cfg_.max_sequence, cfg_.d_model) ||
      !vector_f32(final_weight, cfg_.d_model) || !vector_f32(final_bias, cfg_.d_model))
    return;
  // Embeddings remain F32: they are the external input contract. Projections may use BF16.
  if (!token->is_f32() || !position->is_f32())
    return;
  token_embedding_ = token->as_f32();
  position_embedding_ = position->as_f32();
  final_norm_weight_ = final_weight->as_f32();
  final_norm_bias_ = final_bias->as_f32();
  for (std::size_t index{}; index < cfg_.n_layers; ++index) {
    Layer& layer = layers_[index];
    const TensorView* const ln1_weight = layer_tensor(weights, index, "ln_1.weight");
    const TensorView* const ln1_bias = layer_tensor(weights, index, "ln_1.bias");
    const TensorView* const qkv = layer_tensor(weights, index, "attn.qkv.weight");
    const TensorView* const qkv_bias = layer_tensor(weights, index, "attn.qkv.bias");
    const TensorView* const attention_out = layer_tensor(weights, index, "attn.out_proj.weight");
    const TensorView* const attention_out_bias = layer_tensor(weights, index, "attn.out_proj.bias");
    const TensorView* const ln2_weight = layer_tensor(weights, index, "ln_2.weight");
    const TensorView* const ln2_bias = layer_tensor(weights, index, "ln_2.bias");
    const TensorView* const mlp_in = layer_tensor(weights, index, "mlp.fc_in.weight");
    const TensorView* const mlp_in_bias = layer_tensor(weights, index, "mlp.fc_in.bias");
    const TensorView* const mlp_out = layer_tensor(weights, index, "mlp.fc_out.weight");
    const TensorView* const mlp_out_bias = layer_tensor(weights, index, "mlp.fc_out.bias");
    if (!vector_f32(ln1_weight, cfg_.d_model) || !vector_f32(ln1_bias, cfg_.d_model) ||
        !matrix_weight(qkv, 3uz * cfg_.d_model, cfg_.d_model) ||
        !vector_f32(qkv_bias, 3uz * cfg_.d_model) ||
        !matrix_weight(attention_out, cfg_.d_model, cfg_.d_model) ||
        !vector_f32(attention_out_bias, cfg_.d_model) || !vector_f32(ln2_weight, cfg_.d_model) ||
        !vector_f32(ln2_bias, cfg_.d_model) || !matrix_weight(mlp_in, cfg_.mlp_dim, cfg_.d_model) ||
        !vector_f32(mlp_in_bias, cfg_.mlp_dim) ||
        !matrix_weight(mlp_out, cfg_.d_model, cfg_.mlp_dim) ||
        !vector_f32(mlp_out_bias, cfg_.d_model))
      return;
    layer.ln1_weight = ln1_weight->as_f32();
    layer.ln1_bias = ln1_bias->as_f32();
    layer.qkv = weight(qkv);
    layer.qkv_bias = qkv_bias->as_f32();
    layer.attention_out = weight(attention_out);
    layer.attention_out_bias = attention_out_bias->as_f32();
    layer.ln2_weight = ln2_weight->as_f32();
    layer.ln2_bias = ln2_bias->as_f32();
    layer.mlp_in = weight(mlp_in);
    layer.mlp_in_bias = mlp_in_bias->as_f32();
    layer.mlp_out = weight(mlp_out);
    layer.mlp_out_bias = mlp_out_bias->as_f32();
    layer.key_cache = arena.alloc_array<float, kSimdAlign>(cfg_.max_sequence * cfg_.d_model);
    layer.value_cache = arena.alloc_array<float, kSimdAlign>(cfg_.max_sequence * cfg_.d_model);
    if (layer.key_cache == nullptr || layer.value_cache == nullptr)
      return;
  }
  reset();
  valid_ = true;
}

void Transformer::reset() noexcept
{
  for (std::size_t i{}; i < cfg_.n_layers; ++i) {
    std::fill_n(layers_[i].key_cache, cfg_.max_sequence * cfg_.d_model, 0.0F);
    std::fill_n(layers_[i].value_cache, cfg_.max_sequence * cfg_.d_model, 0.0F);
  }
  position_ = 0uz;
}

bool Transformer::decode_layer(Layer& layer, std::span<float> hidden) noexcept
{
  std::byte* const mark = arena_->mark();
  const std::size_t model = cfg_.d_model;
  const std::size_t head_width = model / cfg_.n_heads;
  std::span<float> normed = scratch(model);
  std::span<float> qkv = scratch(3uz * model);
  std::span<float> attended = scratch(model);
  std::span<float> projected = scratch(model);
  std::span<float> mlp_hidden = scratch(cfg_.mlp_dim);
  std::span<float> mlp_output = scratch(model);
  std::span<float> scores = scratch(position_ + 1uz);
  if (normed.empty() || qkv.empty() || attended.empty() || projected.empty() ||
      mlp_hidden.empty() || mlp_output.empty() || scores.empty()) {
    arena_->reset_to(mark);
    return false;
  }
  layer_norm(hidden, {layer.ln1_weight, model}, {layer.ln1_bias, model}, normed, 1uz, model);
  matmul_weight(normed, layer.qkv, qkv, 1uz, model, 3uz * model, pool_);
  add_inplace(qkv, {layer.qkv_bias, 3uz * model});
  std::copy_n(qkv.data() + model, model, layer.key_cache + (position_ * model));
  std::copy_n(qkv.data() + (2uz * model), model, layer.value_cache + (position_ * model));
  for (std::size_t head{}; head < cfg_.n_heads; ++head) {
    const std::size_t offset = head * head_width;
    const float inverse_scale = 1.0F / std::sqrt(static_cast<float>(head_width));
    for (std::size_t token{}; token <= position_; ++token) {
      const float* const key = layer.key_cache + (token * model) + offset;
      float dot{};
      for (std::size_t channel{}; channel < head_width; ++channel)
        dot += qkv[offset + channel] * key[channel];
      scores[token] = dot * inverse_scale;
    }
    softmax(scores);
    for (std::size_t channel{}; channel < head_width; ++channel) {
      float sum{};
      for (std::size_t token{}; token <= position_; ++token)
        sum += scores[token] * layer.value_cache[(token * model) + offset + channel];
      attended[offset + channel] = sum;
    }
  }
  matmul_weight(attended, layer.attention_out, projected, 1uz, model, model, pool_);
  add_inplace(projected, {layer.attention_out_bias, model});
  add_inplace(hidden, projected);
  layer_norm(hidden, {layer.ln2_weight, model}, {layer.ln2_bias, model}, normed, 1uz, model);
  matmul_weight(normed, layer.mlp_in, mlp_hidden, 1uz, model, cfg_.mlp_dim, pool_);
  add_inplace(mlp_hidden, {layer.mlp_in_bias, cfg_.mlp_dim});
  gelu(mlp_hidden);
  matmul_weight(mlp_hidden, layer.mlp_out, mlp_output, 1uz, cfg_.mlp_dim, model, pool_);
  add_inplace(mlp_output, {layer.mlp_out_bias, model});
  add_inplace(hidden, mlp_output);
  arena_->reset_to(mark);
  return true;
}

bool Transformer::decode_embedding(std::span<const float> embedding,
                                   std::span<float> output) noexcept
{
  if (!valid_ || embedding.size() != cfg_.d_model || output.size() != cfg_.d_model ||
      position_ >= cfg_.max_sequence)
    return false;
  std::byte* const mark = arena_->mark();
  std::span<float> hidden = scratch(cfg_.d_model);
  std::span<float> normalized = scratch(cfg_.d_model);
  if (hidden.empty() || normalized.empty()) {
    arena_->reset_to(mark);
    return false;
  }
  const float* const position = position_embedding_ + (position_ * cfg_.d_model);
  for (std::size_t i{}; i < cfg_.d_model; ++i)
    hidden[i] = embedding[i] + position[i];
  for (std::size_t layer{}; layer < cfg_.n_layers; ++layer)
    if (!decode_layer(layers_[layer], hidden)) {
      arena_->reset_to(mark);
      return false;
    }
  layer_norm(hidden, {final_norm_weight_, cfg_.d_model}, {final_norm_bias_, cfg_.d_model},
             normalized, 1uz, cfg_.d_model);
  copy_span(normalized, output);
  ++position_;
  arena_->reset_to(mark);
  return true;
}

bool Transformer::decode_token(std::int32_t token, std::span<float> output) noexcept
{
  if (!valid_ || token < 0 || static_cast<std::size_t>(token) >= cfg_.vocab)
    return false;
  return decode_embedding({token_embedding_ + (static_cast<std::size_t>(token) * cfg_.d_model),
                           cfg_.d_model},
                          output);
}

bool Transformer::forward_tokens(std::span<const std::int32_t> tokens,
                                 std::span<float> output) noexcept
{
  if (!valid_ || tokens.empty() || tokens.size() > cfg_.max_sequence ||
      output.size() != tokens.size() * cfg_.d_model)
    return false;
  reset();
  for (std::size_t index{}; index < tokens.size(); ++index)
    if (!decode_token(tokens[index], output.subspan(index * cfg_.d_model, cfg_.d_model)))
      return false;
  return true;
}

} // namespace fe
