#include "arena/arena.h"
#include "kernels/kernels.h"
#include "loader/safetensors.h"
#include "models/transformer/transformer.h"
#include "test/test_utils.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <span>
#include <vector>

using fe::test::k_tol;

namespace {

void copy_name(fe::TensorView& view, const char* name)
{
  std::size_t index{};
  for (const char* p = name; (*p != '\0') && (index + 1uz < view.name.size()); ++p)
    view.name[index++] = *p;
}

fe::TensorView view_f32(const char* name, float* data, std::size_t rows, std::size_t cols)
{
  fe::TensorView view{};
  view.data = data;
  view.dtype = fe::TensorView::Dtype::F32;
  view.ndim = 2;
  view.shape[0] = rows;
  view.shape[1] = cols;
  view.bytes = rows * cols * sizeof(float);
  copy_name(view, name);
  return view;
}

fe::TensorView view_f32_1d(const char* name, float* data, std::size_t size)
{
  fe::TensorView view{};
  view.data = data;
  view.dtype = fe::TensorView::Dtype::F32;
  view.ndim = 1;
  view.shape[0] = size;
  view.bytes = size * sizeof(float);
  copy_name(view, name);
  return view;
}

void fill_identity(std::vector<float>& matrix, std::size_t rows, std::size_t cols)
{
  matrix.assign(rows * cols, 0.0F);
  for (std::size_t index{}; index < rows && index < cols; ++index)
    matrix[(index * cols) + index] = 1.0F;
}

void fill_stacked_identity(std::vector<float>& matrix, std::size_t blocks, std::size_t dim)
{
  matrix.assign(blocks * dim * dim, 0.0F);
  for (std::size_t block{}; block < blocks; ++block)
    for (std::size_t index{}; index < dim; ++index)
      matrix[(((block * dim) + index) * dim) + index] = 1.0F;
}

struct TransformerFixture
{
  static constexpr std::size_t kVocab{8uz};
  static constexpr std::size_t kModel{4uz};
  static constexpr std::size_t kLayers{1uz};
  static constexpr std::size_t kHeads{2uz};
  static constexpr std::size_t kMaxSeq{4uz};
  static constexpr std::size_t kMlp{8uz};

  std::vector<float> config_{6uz};
  std::vector<float> token_;
  std::vector<float> position_;
  std::vector<float> ln1_w_{kModel, 1.0F};
  std::vector<float> ln1_b_{kModel, 0.0F};
  std::vector<float> qkv_;
  std::vector<float> qkv_b_{3uz * kModel, 0.0F};
  std::vector<float> attn_out_;
  std::vector<float> attn_out_b_{kModel, 0.0F};
  std::vector<float> ln2_w_{kModel, 1.0F};
  std::vector<float> ln2_b_{kModel, 0.0F};
  std::vector<float> mlp_in_;
  std::vector<float> mlp_in_b_{kMlp, 0.0F};
  std::vector<float> mlp_out_;
  std::vector<float> mlp_out_b_{kModel, 0.0F};
  std::vector<float> final_w_{kModel, 1.0F};
  std::vector<float> final_b_{kModel, 0.0F};
  std::vector<std::byte> slab = std::vector<std::byte>(1uz << 20);
  fe::Arena arena{std::span<std::byte>{slab}};

  TransformerFixture()
  {
    config_ = {static_cast<float>(kVocab), static_cast<float>(kModel),  static_cast<float>(kLayers),
               static_cast<float>(kHeads), static_cast<float>(kMaxSeq), static_cast<float>(kMlp)};
    token_.assign(kVocab * kModel, 0.0F);
    for (std::size_t token{}; token < kVocab; ++token)
      for (std::size_t channel{}; channel < kModel; ++channel)
        token_[(token * kModel) + channel] =
            static_cast<float>(token + 1uz) * 0.1F + static_cast<float>(channel) * 0.01F;
    position_.assign(kMaxSeq * kModel, 0.0F);
    fill_stacked_identity(qkv_, 3uz, kModel);
    fill_identity(attn_out_, kModel, kModel);
    mlp_in_.assign(kMlp * kModel, 0.0F);
    mlp_out_.assign(kModel * kMlp, 0.0F);
  }

  std::vector<fe::TensorView> views()
  {
    return {
        view_f32_1d("transformer.config", config_.data(), config_.size()),
        view_f32("transformer.embeddings.token.weight", token_.data(), kVocab, kModel),
        view_f32("transformer.embeddings.position.weight", position_.data(), kMaxSeq, kModel),
        view_f32_1d("transformer.layers.0.ln_1.weight", ln1_w_.data(), kModel),
        view_f32_1d("transformer.layers.0.ln_1.bias", ln1_b_.data(), kModel),
        view_f32("transformer.layers.0.attn.qkv.weight", qkv_.data(), 3uz * kModel, kModel),
        view_f32_1d("transformer.layers.0.attn.qkv.bias", qkv_b_.data(), 3uz * kModel),
        view_f32("transformer.layers.0.attn.out_proj.weight", attn_out_.data(), kModel, kModel),
        view_f32_1d("transformer.layers.0.attn.out_proj.bias", attn_out_b_.data(), kModel),
        view_f32_1d("transformer.layers.0.ln_2.weight", ln2_w_.data(), kModel),
        view_f32_1d("transformer.layers.0.ln_2.bias", ln2_b_.data(), kModel),
        view_f32("transformer.layers.0.mlp.fc_in.weight", mlp_in_.data(), kMlp, kModel),
        view_f32_1d("transformer.layers.0.mlp.fc_in.bias", mlp_in_b_.data(), kMlp),
        view_f32("transformer.layers.0.mlp.fc_out.weight", mlp_out_.data(), kModel, kMlp),
        view_f32_1d("transformer.layers.0.mlp.fc_out.bias", mlp_out_b_.data(), kModel),
        view_f32_1d("transformer.norm_f.weight", final_w_.data(), kModel),
        view_f32_1d("transformer.norm_f.bias", final_b_.data(), kModel),
    };
  }
};

} // namespace

TEST(Gelu, MatchesTanhFormula)
{
  std::vector<float> values{-2.0F, -0.5F, 0.0F, 0.25F, 1.5F};
  const std::vector<float> input = values;
  fe::gelu(values);
  constexpr float kSqrtTwoOverPi = 0.7978845608028654F;
  constexpr float kCoefficient = 0.044715F;
  for (std::size_t index{}; index < input.size(); ++index) {
    const float x = input[index];
    const float expected =
        0.5F * x * (1.0F + std::tanh(kSqrtTwoOverPi * (x + (kCoefficient * x * x * x))));
    EXPECT_NEAR(values[index], expected, 1e-6F);
  }
}

TEST(LayerNorm, AffineOverRows)
{
  const std::vector<float> input{1.0F, 3.0F, 5.0F, 7.0F};
  const std::vector<float> weight{2.0F, 0.5F};
  const std::vector<float> bias{-1.0F, 0.25F};
  std::vector<float> out(4uz);
  fe::layer_norm(input, weight, bias, out, 2uz, 2uz);
  for (std::size_t row{}; row < 2uz; ++row) {
    const float a = input[row * 2uz];
    const float b = input[(row * 2uz) + 1uz];
    const float mean = 0.5F * (a + b);
    const float var = 0.5F * ((a - mean) * (a - mean) + (b - mean) * (b - mean));
    const float scale = 1.0F / std::sqrt(var + 1e-5F);
    EXPECT_NEAR(out[row * 2uz], ((a - mean) * scale * weight[0]) + bias[0], 1e-5F);
    EXPECT_NEAR(out[(row * 2uz) + 1uz], ((b - mean) * scale * weight[1]) + bias[1], 1e-5F);
  }
}

TEST(Softmax, SumsToOne)
{
  std::vector<float> values{1.0F, 2.0F, 3.0F};
  fe::softmax(values);
  float sum{};
  for (float value : values)
    sum += value;
  EXPECT_NEAR(sum, 1.0F, 1e-6F);
  EXPECT_GT(values[2], values[1]);
  EXPECT_GT(values[1], values[0]);
}

TEST(CachedCausalAttention, SingleTokenCopiesValue)
{
  constexpr std::size_t kModel{4uz};
  const std::vector<float> query{0.5F, -0.25F, 1.0F, 0.0F};
  std::vector<float> keys = query;
  std::vector<float> values = query;
  std::vector<float> scores(1uz);
  std::vector<float> attended(kModel);
  fe::cached_causal_attention(query, keys, values, {}, scores, attended, 2uz, kModel, 0uz);
  for (std::size_t index{}; index < kModel; ++index)
    EXPECT_NEAR(attended[index], query[index], 1e-5F);
}

TEST(CachedCausalAttention, MaskDropsPaddingToken)
{
  constexpr std::size_t kModel{2uz};
  const std::vector<float> query{1.0F, 0.0F};
  const std::vector<float> keys{1.0F, 0.0F, 0.0F, 1.0F};
  const std::vector<float> values{2.0F, 0.0F, 9.0F, 9.0F};
  const std::vector<std::uint8_t> mask{1u, 0u};
  std::vector<float> scores(2uz);
  std::vector<float> attended(kModel);
  fe::cached_causal_attention(query, keys, values, mask, scores, attended, 1uz, kModel, 1uz);
  EXPECT_NEAR(attended[0], 2.0F, 1e-5F);
  EXPECT_NEAR(attended[1], 0.0F, 1e-5F);
}

TEST(Transformer, LoadsFixtureAndResetsKv)
{
  TransformerFixture fixture;
  const auto views = fixture.views();
  fe::Transformer model{views, fixture.arena};
  ASSERT_TRUE(model.valid());
  EXPECT_EQ(model.config().max_sequence, TransformerFixture::kMaxSeq);
  EXPECT_EQ(model.position(), 0uz);
  std::vector<float> hidden(TransformerFixture::kModel);
  ASSERT_TRUE(model.decode_token(1, hidden));
  EXPECT_EQ(model.position(), 1uz);
  model.reset();
  EXPECT_EQ(model.position(), 0uz);
}

TEST(Transformer, RejectsPastMaxPrefix)
{
  TransformerFixture fixture;
  const auto views = fixture.views();
  fe::Transformer model{views, fixture.arena};
  ASSERT_TRUE(model.valid());
  std::vector<float> hidden(TransformerFixture::kModel);
  for (std::size_t step{}; step < TransformerFixture::kMaxSeq; ++step)
    ASSERT_TRUE(model.decode_token(static_cast<std::int32_t>(step), hidden));
  EXPECT_FALSE(model.decode_token(0, hidden));
}

TEST(Transformer, PrefixMatchesStreaming)
{
  TransformerFixture fixture;
  const auto views = fixture.views();
  fe::Transformer model{views, fixture.arena};
  ASSERT_TRUE(model.valid());
  const std::array<std::int32_t, 3> tokens{1, 2, 3};
  std::vector<float> prefix(tokens.size() * TransformerFixture::kModel);
  ASSERT_TRUE(model.forward_tokens(tokens, prefix));
  model.reset();
  std::vector<float> streamed(prefix.size());
  for (std::size_t index{}; index < tokens.size(); ++index) {
    std::span<float> row{streamed.data() + (index * TransformerFixture::kModel),
                         TransformerFixture::kModel};
    ASSERT_TRUE(model.decode_token(tokens[index], row));
  }
  for (std::size_t index{}; index < prefix.size(); ++index)
    EXPECT_NEAR(prefix[index], streamed[index], k_tol);
}

TEST(Transformer, EmbeddingMaskRejectsPaddingThenValid)
{
  TransformerFixture fixture;
  const auto views = fixture.views();
  fe::Transformer model{views, fixture.arena};
  ASSERT_TRUE(model.valid());
  std::vector<float> embeddings(2uz * TransformerFixture::kModel, 0.1F);
  std::vector<float> output(embeddings.size());
  const std::vector<std::uint8_t> bad{0u, 1u};
  EXPECT_FALSE(model.forward_embeddings(embeddings, bad, output));
  const std::vector<std::uint8_t> ok{1u, 0u};
  EXPECT_TRUE(model.forward_embeddings(embeddings, ok, output));
}
