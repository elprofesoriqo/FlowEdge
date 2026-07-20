#include "mamba.h"

#include "cpu/kernels.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <string_view>

namespace fe {
namespace {

const TensorView* find(std::span<const TensorView> ts, std::string_view name) noexcept
{
  for (const auto& t : ts)
    if (t.name_view() == name)
      return &t;
  return nullptr;
}

std::string_view layer_key(std::span<char> buf, std::size_t i, std::string_view sub) noexcept
{
  std::size_t p{0uz};
  const auto put = [&](std::string_view s) noexcept {
    for (const char ch : s)
      if (p + 1uz < buf.size())
        buf[p++] = ch;
  };
  put("backbone.layers.");
  std::array<char, 20> digits{};
  std::size_t nd{0uz};
  for (; i != 0uz; i /= 10uz)
    digits[nd++] = static_cast<char>('0' + (i % 10uz));
  if (nd == 0uz)
    digits[nd++] = '0';
  while (nd > 0uz)
    put({&digits[--nd], 1uz});
  put(".");
  put(sub);
  buf[p] = '\0';
  return {buf.data(), p};
}

const float* layer_weight(std::span<const TensorView> ts, std::size_t i, std::string_view sub,
                          bool& ok) noexcept
{
  std::array<char, 96> buf{};
  const TensorView* t = find(ts, layer_key(buf, i, sub));
  ok = ok && (t != nullptr);
  return (t != nullptr) ? t->data : nullptr;
}

} // namespace

Mamba::Mamba(std::span<const TensorView> weights, Arena& scratch) noexcept : scratch_{&scratch}
{
  const TensorView* emb = find(weights, "backbone.embeddings.weight");
  const TensorView* a0 = find(weights, "backbone.layers.0.mixer.A_log");
  const TensorView* cv0 = find(weights, "backbone.layers.0.mixer.conv1d.weight");
  const TensorView* xp0 = find(weights, "backbone.layers.0.mixer.x_proj.weight");
  const TensorView* nf = find(weights, "backbone.norm_f.weight");
  if ((emb == nullptr) || (a0 == nullptr) || (cv0 == nullptr) || (xp0 == nullptr) ||
      (nf == nullptr))
    return;

  cfg_.vocab = emb->shape[0];
  cfg_.d_model = emb->shape[1];
  cfg_.d_inner = a0->shape[0];
  cfg_.d_state = a0->shape[1];
  cfg_.d_conv = cv0->shape[2];
  cfg_.dt_rank = xp0->shape[0] - (2uz * cfg_.d_state);
  emb_ = emb->data;
  norm_f_ = nf->data;

  std::size_t n{0uz};
  for (; n < kMaxLayers; ++n) {
    bool present{true};
    const float* norm = layer_weight(weights, n, "norm.weight", present);
    if (!present)
      break;
    Layer& lw = layers_[n];
    lw.norm = norm;
    bool lok{true};
    lw.in_proj = layer_weight(weights, n, "mixer.in_proj.weight", lok);
    lw.conv_w = layer_weight(weights, n, "mixer.conv1d.weight", lok);
    lw.conv_b = layer_weight(weights, n, "mixer.conv1d.bias", lok);
    lw.x_proj = layer_weight(weights, n, "mixer.x_proj.weight", lok);
    lw.dt_w = layer_weight(weights, n, "mixer.dt_proj.weight", lok);
    lw.dt_b = layer_weight(weights, n, "mixer.dt_proj.bias", lok);
    lw.a_log = layer_weight(weights, n, "mixer.A_log", lok);
    lw.d = layer_weight(weights, n, "mixer.D", lok);
    lw.out_proj = layer_weight(weights, n, "mixer.out_proj.weight", lok);
    if (!lok)
      return; // malformed layer
  }
  cfg_.n_layers = n;
  ok_ = n > 0uz;
}

void Mamba::layer_forward(const Layer& lw, std::span<float> hidden, std::size_t seq_len) noexcept
{
  const std::size_t dm = cfg_.d_model;
  const std::size_t di = cfg_.d_inner;
  const std::size_t ds = cfg_.d_state;
  const std::size_t dr = cfg_.dt_rank;
  const std::size_t wd = dr + (2uz * ds); // x_proj output width: [dt | B | C]
  const std::size_t l = seq_len;

  std::byte* const mark = scratch_->mark();
  const auto buf = [&](std::size_t count) noexcept {
    return std::span<float>{scratch_->alloc_array<float>(count, kSimdAlign), count};
  };
  const std::span<float> normed = buf(l * dm);
  const std::span<float> xz = buf(l * 2uz * di);
  const std::span<float> z = buf(l * di);
  const std::span<float> x_cm = buf(di * l); // channel-major for conv
  const std::span<float> x_sm = buf(l * di); // seq-major conv+silu output = scan input u
  const std::span<float> dbl = buf(l * wd);
  const std::span<float> dt = buf(l * di);
  const std::span<float> c_buf = buf(l * ds);
  const std::span<float> da = buf(l * ds * di);
  const std::span<float> dbu = buf(l * ds * di);
  const std::span<float> h = buf(ds * di);
  const std::span<float> yv = buf(l * di);
  const std::span<float> out = buf(l * dm);

  rmsnorm(hidden, {lw.norm, dm}, normed, l, dm);
  matmul(normed, {lw.in_proj, 2uz * di * dm}, xz, l, dm, 2uz * di); // [l][x|z]

  for (std::size_t t{0uz}; t < l; ++t)
    for (std::size_t c{0uz}; c < di; ++c) {
      x_cm[(c * l) + t] = xz[(t * 2uz * di) + c];
      z[(t * di) + c] = xz[(t * 2uz * di) + di + c];
    }

  conv1d_causal(x_cm, {lw.conv_w, di * cfg_.d_conv}, {lw.conv_b, di}, x_cm, di, l, cfg_.d_conv);
  silu(x_cm);
  for (std::size_t t{0uz}; t < l; ++t)
    for (std::size_t c{0uz}; c < di; ++c)
      x_sm[(t * di) + c] = x_cm[(c * l) + t];

  matmul(x_sm, {lw.x_proj, wd * di}, dbl, l, di, wd); // [l][dt | B | C]

  for (std::size_t t{0uz}; t < l; ++t)
    for (std::size_t o{0uz}; o < di; ++o) {
      float acc = lw.dt_b[o];
      for (std::size_t k{0uz}; k < dr; ++k)
        acc += dbl[(t * wd) + k] * lw.dt_w[(o * dr) + k];
      dt[(t * di) + o] = acc;
    }
  softplus(dt);

  for (std::size_t t{0uz}; t < l; ++t)
    for (std::size_t nn{0uz}; nn < ds; ++nn)
      c_buf[(t * ds) + nn] = dbl[(t * wd) + dr + ds + nn];

  // discretize: A = exp(Δ·A), Bu = Δ·B·u, state-major [t][n][c] for the scan.
  for (std::size_t t{0uz}; t < l; ++t)
    for (std::size_t nn{0uz}; nn < ds; ++nn)
      for (std::size_t c{0uz}; c < di; ++c) {
        const float a = -std::exp(lw.a_log[(c * ds) + nn]);
        const float dtc = dt[(t * di) + c];
        const std::size_t idx = ((t * ds) + nn) * di + c;
        da[idx] = std::exp(dtc * a);
        dbu[idx] = dtc * dbl[(t * wd) + dr + nn] * x_sm[(t * di) + c];
      }

  selective_scan(da, dbu, c_buf, {lw.d, di}, x_sm, h, yv, l, di, ds);
  gate_silu(yv, z, yv); // y · silu(z)
  matmul(yv, {lw.out_proj, dm * di}, out, l, di, dm);

  for (std::size_t i{0uz}; i < l * dm; ++i)
    hidden[i] += out[i]; // residual

  scratch_->reset_to(mark);
}

void Mamba::forward(std::span<const float> input, std::span<float> output,
                    std::size_t seq_len) noexcept
{
  const std::size_t hz = seq_len * cfg_.d_model;
  for (std::size_t i{0uz}; i < hz; ++i)
    output[i] = input[i];
  for (std::size_t layer{0uz}; layer < cfg_.n_layers; ++layer)
    layer_forward(layers_[layer], output.first(hz), seq_len);
  rmsnorm(output.first(hz), {norm_f_, cfg_.d_model}, output.first(hz), seq_len, cfg_.d_model);
}

} // namespace fe
