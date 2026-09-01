#include "mamba.h"

#include "kernels/kernels.h"
#include "kernels/span_ops.h"
#include "loader/tensor_key.h"
#include "loader/weight_ops.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <span>
#include <string_view>

#if __has_include(<mdspan>)
#include <mdspan>
namespace fe_md = std;
#else
#include <experimental/mdspan>
namespace fe_md = std::experimental;
#endif

namespace fe {
namespace {

[[nodiscard]] bool is_vector_f32(const TensorView* tensor, std::size_t size) noexcept
{
  return tensor != nullptr && tensor->ndim == 1uz && tensor->shape[0] == size && tensor->is_f32();
}

[[nodiscard]] bool is_matrix_weight(const TensorView* tensor, std::size_t rows,
                                    std::size_t columns) noexcept
{
  return tensor != nullptr && tensor->ndim == 2uz && tensor->shape[0] == rows &&
         tensor->shape[1] == columns && is_matmul_weight(tensor);
}

[[nodiscard]] bool is_conv_weight(const TensorView* tensor, std::size_t channels,
                                  std::size_t kernel) noexcept
{
  return tensor != nullptr && tensor->ndim == 3uz && tensor->shape[0] == channels &&
         tensor->shape[1] == 1uz && tensor->shape[2] == kernel && tensor->is_f32();
}

std::string_view layer_key(std::span<char> buf, std::size_t i, std::string_view sub) noexcept
{
  TensorKeyBuilder key{buf};
  if (!key.append("backbone.layers.") || !key.append(i) || !key.append(".") || !key.append(sub))
    return {};
  return key.view();
}

const TensorView* layer_tensor(std::span<const TensorView> ts, std::size_t i,
                               std::string_view sub) noexcept
{
  std::array<char, 96> buf{};
  return find_tensor(ts, layer_key(buf, i, sub));
}

const float* layer_weight_f32(std::span<const TensorView> ts, std::size_t i, std::string_view sub,
                              bool& ok) noexcept
{
  const TensorView* t = layer_tensor(ts, i, sub);
  ok = ok && (t != nullptr) && t->is_f32();
  return (t != nullptr) ? t->as_f32() : nullptr;
}

WeightView layer_weight(std::span<const TensorView> ts, std::size_t i, std::string_view sub,
                        bool& ok) noexcept
{
  const TensorView* t = layer_tensor(ts, i, sub);
  ok = ok && is_matmul_weight(t);
  return weight_view(t);
}

} // namespace

Mamba::Mamba(std::span<const TensorView> weights, Arena& scratch) noexcept : scratch_{&scratch}
{
  const TensorView* emb = find_tensor(weights, "backbone.embeddings.weight");
  const TensorView* a0 = find_tensor(weights, "backbone.layers.0.mixer.A_log");
  const TensorView* cv0 = find_tensor(weights, "backbone.layers.0.mixer.conv1d.weight");
  const TensorView* xp0 = find_tensor(weights, "backbone.layers.0.mixer.x_proj.weight");
  const TensorView* nf = find_tensor(weights, "backbone.norm_f.weight");
  if ((emb == nullptr) || (a0 == nullptr) || (cv0 == nullptr) || (xp0 == nullptr) ||
      (nf == nullptr) || emb->ndim != 2uz || a0->ndim != 2uz || cv0->ndim != 3uz ||
      xp0->ndim != 2uz)
    return;

  cfg_.vocab = emb->shape[0];
  cfg_.d_model = emb->shape[1];
  cfg_.d_inner = a0->shape[0];
  cfg_.d_state = a0->shape[1];
  cfg_.d_conv = cv0->shape[2];
  if (cfg_.vocab == 0uz || cfg_.d_model == 0uz || cfg_.d_inner == 0uz || cfg_.d_state == 0uz ||
      cfg_.d_conv == 0uz || xp0->shape[0] < 2uz * cfg_.d_state)
    return;
  cfg_.dt_rank = xp0->shape[0] - (2uz * cfg_.d_state);
  if (cfg_.dt_rank == 0uz || !emb->is_f32() || !a0->is_f32() || !is_vector_f32(nf, cfg_.d_model) ||
      !is_conv_weight(cv0, cfg_.d_inner, cfg_.d_conv) ||
      !is_matrix_weight(xp0, cfg_.dt_rank + (2uz * cfg_.d_state), cfg_.d_inner))
    return;
  emb_ = emb->as_f32();
  norm_f_ = nf->as_f32();

  std::size_t n{0uz};
  for (; n < kMaxLayers; ++n) {
    bool present{true};
    const TensorView* norm = layer_tensor(weights, n, "norm.weight");
    present = norm != nullptr;
    if (!present)
      break;
    Layer& lw = layers_[n];
    bool lok{true};
    lw.norm = layer_weight_f32(weights, n, "norm.weight", lok);
    lw.in_proj = layer_weight(weights, n, "mixer.in_proj.weight", lok);
    lw.conv_w = layer_weight_f32(weights, n, "mixer.conv1d.weight", lok);
    lw.conv_b = layer_weight_f32(weights, n, "mixer.conv1d.bias", lok);
    lw.x_proj = layer_weight(weights, n, "mixer.x_proj.weight", lok);
    lw.dt_w = layer_weight(weights, n, "mixer.dt_proj.weight", lok);
    lw.dt_b = layer_weight_f32(weights, n, "mixer.dt_proj.bias", lok);
    const float* const a_log = layer_weight_f32(weights, n, "mixer.A_log", lok);
    lw.d = layer_weight_f32(weights, n, "mixer.D", lok);
    lw.out_proj = layer_weight(weights, n, "mixer.out_proj.weight", lok);
    if (!lok || !is_vector_f32(layer_tensor(weights, n, "norm.weight"), cfg_.d_model) ||
        !is_matrix_weight(layer_tensor(weights, n, "mixer.in_proj.weight"), 2uz * cfg_.d_inner,
                          cfg_.d_model) ||
        !is_conv_weight(layer_tensor(weights, n, "mixer.conv1d.weight"), cfg_.d_inner,
                        cfg_.d_conv) ||
        !is_vector_f32(layer_tensor(weights, n, "mixer.conv1d.bias"), cfg_.d_inner) ||
        !is_matrix_weight(layer_tensor(weights, n, "mixer.x_proj.weight"),
                          cfg_.dt_rank + (2uz * cfg_.d_state), cfg_.d_inner) ||
        !is_matrix_weight(layer_tensor(weights, n, "mixer.dt_proj.weight"), cfg_.d_inner,
                          cfg_.dt_rank) ||
        !is_vector_f32(layer_tensor(weights, n, "mixer.dt_proj.bias"), cfg_.d_inner) ||
        !is_matrix_weight(layer_tensor(weights, n, "mixer.A_log"), cfg_.d_inner, cfg_.d_state) ||
        !is_vector_f32(layer_tensor(weights, n, "mixer.D"), cfg_.d_inner) ||
        !is_matrix_weight(layer_tensor(weights, n, "mixer.out_proj.weight"), cfg_.d_model,
                          cfg_.d_inner))
      return; // malformed layer

    // A_log is immutable. Transform and transpose it once so every scan can
    // stream contiguous channels without repeating exp() or a layout pass.
    auto* const a_neg = scratch.alloc_array<float, kSimdAlign>(cfg_.d_state * cfg_.d_inner);
    if (a_neg == nullptr)
      return;
    for (std::size_t nn{0uz}; nn < cfg_.d_state; ++nn)
      for (std::size_t c{0uz}; c < cfg_.d_inner; ++c)
        a_neg[(nn * cfg_.d_inner) + c] = -std::exp(a_log[(c * cfg_.d_state) + nn]);
    lw.a_neg = a_neg;
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

  fe_md::mdspan x_cm_md{x_cm.data(), di, l};
  fe_md::mdspan xz_md{xz.data(), l, 2uz * di};
  fe_md::mdspan z_md{z.data(), l, di};
  fe_md::mdspan x_conv_md{x_conv.data(), di, l};
  fe_md::mdspan x_sm_md{x_sm.data(), l, di};
  fe_md::mdspan dbl_md{dbl.data(), l, wd};
  fe_md::mdspan dt_in_md{dt_in.data(), l, dr};
  fe_md::mdspan dt_md{dt.data(), l, di};
  fe_md::mdspan b_buf_md{b_buf.data(), l, ds};
  fe_md::mdspan c_buf_md{c_buf.data(), l, ds};

  const std::span<float> h = arena_span(ds * di);
  const std::span<float> yv = arena_span(l * di);
  const std::span<float> out = arena_span(l * dm);

  rmsnorm(hidden, {lw.norm, dm}, normed, l, dm);
  matmul_weight(normed, lw.in_proj, xz, l, dm, 2uz * di, pool_); // [l][x|z]

  for (std::size_t t{0uz}; t < l; ++t)
    for (std::size_t c{0uz}; c < di; ++c) {
      x_cm_md[c, t] = xz_md[t, c];
      z_md[t, c] = xz_md[t, di + c];
    }

  conv1d_causal(x_cm, {lw.conv_w, di * cfg_.d_conv}, {lw.conv_b, di}, x_conv, di, l, cfg_.d_conv);
  silu(x_conv);
  for (std::size_t t{0uz}; t < l; ++t)
    for (std::size_t c{0uz}; c < di; ++c)
      x_sm_md[t, c] = x_conv_md[c, t];

  matmul_weight(x_sm, lw.x_proj, dbl, l, di, wd); // [l][dt | B | C]

  // dt = softplus(dt_proj · dbl[:, :dt_rank] + bias); gather the dt slice for matmul
  for (std::size_t t{0uz}; t < l; ++t)
    for (std::size_t k{0uz}; k < dr; ++k)
      dt_in_md[t, k] = dbl_md[t, k];
  matmul_weight(dt_in, lw.dt_w, dt, l, dr, di);
  for (std::size_t t{0uz}; t < l; ++t)
    for (std::size_t o{0uz}; o < di; ++o)
      dt_md[t, o] += lw.dt_b[o];
  softplus(dt);

  for (std::size_t t{0uz}; t < l; ++t)
    for (std::size_t nn{0uz}; nn < ds; ++nn) {
      b_buf_md[t, nn] = dbl_md[t, dr + nn];
      c_buf_md[t, nn] = dbl_md[t, dr + ds + nn];
    }

  discretize_and_scan(dt, {lw.a_neg, di * ds}, b_buf, x_sm, c_buf, {lw.d, di}, h, yv, l, di, ds,
                      true);
  gate_silu(yv, z, yv); // y · silu(z)
  matmul_weight(yv, lw.out_proj, out, l, di, dm, pool_);

  add_inplace(hidden, out); // residual

  scratch_->reset_to(mark);
}

void Mamba::forward(std::span<const float> input, std::span<float> output,
                    std::size_t seq_len) noexcept
{
  const std::size_t hz = seq_len * cfg_.d_model;
  copy_span(input.first(hz), output.first(hz));
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
  const std::span<float> yv = arena_span(di);
  const std::span<float> out = arena_span(dm);

  rmsnorm(hidden, {lw.norm, dm}, normed, 1uz, dm);
  matmul_weight(normed, lw.in_proj, xz, 1uz, dm, 2uz * di); // [x|z]

  for (std::size_t c{0uz}; c < di; ++c) {
    float* const w = conv_win.data() + (c * dc);
    std::memmove(w, w + 1uz, (dc - 1uz) * sizeof(float)); // shift window
    w[dc - 1uz] = xz[c];                                  // newest conv input
    z[c] = xz[di + c];
  }
  conv1d_step(conv_win, {lw.conv_w, di * dc}, {lw.conv_b, di}, x_conv, di, dc);
  silu(x_conv);

  matmul_weight(x_conv, lw.x_proj, dbl, 1uz, di, wd); // [dt | B | C]
  for (std::size_t k{0uz}; k < dr; ++k)
    dt_in[k] = dbl[k];
  matmul_weight(dt_in, lw.dt_w, dt, 1uz, dr, di);
  for (std::size_t o{0uz}; o < di; ++o)
    dt[o] += lw.dt_b[o];
  softplus(dt);
  for (std::size_t nn{0uz}; nn < ds; ++nn) {
    b_buf[nn] = dbl[dr + nn];
    c_buf[nn] = dbl[dr + ds + nn];
  }

  discretize_and_scan(dt, {lw.a_neg, di * ds}, b_buf, x_conv, c_buf, {lw.d, di}, h, yv, 1uz, di, ds,
                      false);
  gate_silu(yv, z, yv);
  matmul_weight(yv, lw.out_proj, out, 1uz, di, dm);

  add_inplace(hidden, out); // residual

  scratch_->reset_to(mark);
}

void Mamba::decode(std::span<const float> x, std::span<float> state, std::span<float> out) noexcept
{
  const std::size_t dm = cfg_.d_model;
  copy_span(x.first(dm), out.first(dm));
  const std::size_t per = cfg_.d_inner * (cfg_.d_conv + cfg_.d_state);
  for (std::size_t layer{0uz}; layer < cfg_.n_layers; ++layer)
    decode_layer(layers_[layer], out.first(dm), state.subspan(layer * per, per));
  rmsnorm(out.first(dm), {norm_f_, dm}, out.first(dm), 1uz, dm);
}

} // namespace fe
