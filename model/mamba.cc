#include "mamba.h"

#include "cpu/kernels.h"

#include <array>
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
  const std::span<float> normed = arena_span(l * dm);
  const std::span<float> xz = arena_span(l * 2uz * di);
  const std::span<float> z = arena_span(l * di);
  const std::span<float> x_cm = arena_span(di * l); // channel-major conv input
  const std::span<float> x_conv =
      arena_span(di * l);                           // conv output distinct (kernel forbids overlap)
  const std::span<float> x_sm = arena_span(l * di); // seq-major conv+silu output = scan input u
  const std::span<float> dbl = arena_span(l * wd);
  const std::span<float> dt_in = arena_span(l * dr); // gathered dt slice of dbl
  const std::span<float> dt = arena_span(l * di);
  const std::span<float> c_buf = arena_span(l * ds);
  const std::span<float> b_buf = arena_span(l * ds);
  const std::span<float> da = arena_span(l * ds * di);
  const std::span<float> dbu = arena_span(l * ds * di);
  const std::span<float> a_neg = arena_span(ds * di); // transposed A scratch for discretize
  const std::span<float> h = arena_span(ds * di);
  const std::span<float> yv = arena_span(l * di);
  const std::span<float> out = arena_span(l * dm);

  rmsnorm(hidden, {lw.norm, dm}, normed, l, dm);
  matmul(normed, {lw.in_proj, 2uz * di * dm}, xz, l, dm, 2uz * di); // [l][x|z]

  for (std::size_t t{0uz}; t < l; ++t)
    for (std::size_t c{0uz}; c < di; ++c) {
      x_cm[(c * l) + t] = xz[(t * 2uz * di) + c];
      z[(t * di) + c] = xz[(t * 2uz * di) + di + c];
    }

  conv1d_causal(x_cm, {lw.conv_w, di * cfg_.d_conv}, {lw.conv_b, di}, x_conv, di, l, cfg_.d_conv);
  silu(x_conv);
  for (std::size_t t{0uz}; t < l; ++t)
    for (std::size_t c{0uz}; c < di; ++c)
      x_sm[(t * di) + c] = x_conv[(c * l) + t];

  matmul(x_sm, {lw.x_proj, wd * di}, dbl, l, di, wd); // [l][dt | B | C]

  // dt = softplus(dt_proj · dbl[:, :dt_rank] + bias); gather the dt slice for matmul
  for (std::size_t t{0uz}; t < l; ++t)
    for (std::size_t k{0uz}; k < dr; ++k)
      dt_in[(t * dr) + k] = dbl[(t * wd) + k];
  matmul(dt_in, {lw.dt_w, di * dr}, dt, l, dr, di);
  for (std::size_t t{0uz}; t < l; ++t)
    for (std::size_t o{0uz}; o < di; ++o)
      dt[(t * di) + o] += lw.dt_b[o];
  softplus(dt);

  for (std::size_t t{0uz}; t < l; ++t)
    for (std::size_t nn{0uz}; nn < ds; ++nn) {
      b_buf[(t * ds) + nn] = dbl[(t * wd) + dr + nn];
      c_buf[(t * ds) + nn] = dbl[(t * wd) + dr + ds + nn];
    }

  discretize(dt, {lw.a_log, di * ds}, b_buf, x_sm, da, dbu, a_neg, l, di, ds);
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

void Mamba::decode_layer(const Layer& lw, std::span<float> hidden, std::span<float> lstate) noexcept
{
  const std::size_t dm = cfg_.d_model;
  const std::size_t di = cfg_.d_inner;
  const std::size_t ds = cfg_.d_state;
  const std::size_t dr = cfg_.dt_rank;
  const std::size_t dc = cfg_.d_conv;
  const std::size_t wd = dr + (2uz * ds);
  const std::span<float> conv_win = lstate.first(di * dc);     // [di][dc], newest at dc-1
  const std::span<float> h = lstate.subspan(di * dc, ds * di); // persistent SSM state

  std::byte* const mark = scratch_->mark();
  const std::span<float> normed = arena_span(dm);
  const std::span<float> xz = arena_span(2uz * di);
  const std::span<float> z = arena_span(di);
  const std::span<float> x_conv = arena_span(di);
  const std::span<float> dbl = arena_span(wd);
  const std::span<float> dt_in = arena_span(dr);
  const std::span<float> dt = arena_span(di);
  const std::span<float> b_buf = arena_span(ds);
  const std::span<float> c_buf = arena_span(ds);
  const std::span<float> da = arena_span(ds * di);
  const std::span<float> dbu = arena_span(ds * di);
  const std::span<float> a_neg = arena_span(ds * di);
  const std::span<float> yv = arena_span(di);
  const std::span<float> out = arena_span(dm);

  rmsnorm(hidden, {lw.norm, dm}, normed, 1uz, dm);
  matmul(normed, {lw.in_proj, 2uz * di * dm}, xz, 1uz, dm, 2uz * di); // [x|z]

  for (std::size_t c{0uz}; c < di; ++c) {
    float* const w = conv_win.data() + (c * dc);
    for (std::size_t k{0uz}; k + 1uz < dc; ++k)
      w[k] = w[k + 1uz]; // drop oldest, shift window
    w[dc - 1uz] = xz[c]; // newest conv input
    z[c] = xz[di + c];
  }
  conv1d_step(conv_win, {lw.conv_w, di * dc}, {lw.conv_b, di}, x_conv, di, dc);
  silu(x_conv);

  matmul(x_conv, {lw.x_proj, wd * di}, dbl, 1uz, di, wd); // [dt | B | C]
  for (std::size_t k{0uz}; k < dr; ++k)
    dt_in[k] = dbl[k];
  matmul(dt_in, {lw.dt_w, di * dr}, dt, 1uz, dr, di);
  for (std::size_t o{0uz}; o < di; ++o)
    dt[o] += lw.dt_b[o];
  softplus(dt);
  for (std::size_t nn{0uz}; nn < ds; ++nn) {
    b_buf[nn] = dbl[dr + nn];
    c_buf[nn] = dbl[dr + ds + nn];
  }

  discretize(dt, {lw.a_log, di * ds}, b_buf, x_conv, da, dbu, a_neg, 1uz, di, ds);
  scan_step(da, dbu, c_buf, {lw.d, di}, x_conv, h, yv, di, ds);
  gate_silu(yv, z, yv);
  matmul(yv, {lw.out_proj, dm * di}, out, 1uz, di, dm);

  for (std::size_t i{0uz}; i < dm; ++i)
    hidden[i] += out[i]; // residual

  scratch_->reset_to(mark);
}

void Mamba::decode(std::span<const float> x, std::span<float> state, std::span<float> out) noexcept
{
  const std::size_t dm = cfg_.d_model;
  for (std::size_t i{0uz}; i < dm; ++i)
    out[i] = x[i];
  const std::size_t per = cfg_.d_inner * (cfg_.d_conv + cfg_.d_state);
  for (std::size_t layer{0uz}; layer < cfg_.n_layers; ++layer)
    decode_layer(layers_[layer], out.first(dm), state.subspan(layer * per, per));
  rmsnorm(out.first(dm), {norm_f_, dm}, out.first(dm), 1uz, dm);
}

} // namespace fe