#include "flow.h"

#include "../cpu/kernels.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <string_view>
#include <utility>

namespace fe {
namespace {

const TensorView* find(std::span<const TensorView> ts, std::string_view name) noexcept
{
  for (const auto& t : ts)
    if (t.name_view() == name)
      return &t;
  return nullptr;
}

} // namespace

FlowHead::FlowHead(std::span<const TensorView> weights, Arena& scratch) noexcept
    : scratch_{&scratch}
{
  const TensorView* in = find(weights, "flow.in_proj.weight");
  const TensorView* tp = find(weights, "flow.time_proj.weight");
  const TensorView* cp = find(weights, "flow.cond_proj.weight");
  const TensorView* op = find(weights, "flow.out_proj.weight");
  if ((in == nullptr) || (tp == nullptr) || (cp == nullptr) || (op == nullptr))
    return;

  cfg_.hidden = in->shape[0];
  cfg_.action_dim = in->shape[1];
  cfg_.time_dim = tp->shape[1];
  cfg_.cond_dim = cp->shape[1];
  in_proj_ = in->data;
  time_proj_ = tp->data;
  cond_proj_ = cp->data;
  out_proj_ = op->data;

  std::size_t n{0uz};
  for (; n < kMaxMlp; ++n) {
    std::array<char, 48> buf{};
    std::size_t p{0uz};
    for (const char ch : std::string_view{"flow.layers."})
      buf[p++] = ch;
    buf[p++] = static_cast<char>('0' + n); // mlp_layers < 10
    for (const char ch : std::string_view{".weight"})
      buf[p++] = ch;
    const TensorView* lw = find(weights, {buf.data(), p});
    if (lw == nullptr)
      break;
    layers_[n] = lw->data;
  }
  cfg_.mlp_layers = n;
  if (cfg_.time_dim % 2uz != 0uz) // sinusoidal embed needs sin/cos pairs
    return;

  const std::size_t half = cfg_.time_dim / 2uz; // sinusoidal freqs are constant
  auto* const f = scratch.alloc_array<float>(half, kSimdAlign);
  if (f == nullptr)
    return;
  for (std::size_t j{0uz}; j < half; ++j)
    f[j] = std::pow(10000.0F, -static_cast<float>(j) / static_cast<float>(half));
  freqs_ = f;
  ok_ = true;
}

void FlowHead::velocity(std::span<const float> x, float t, std::span<const float> c_emb,
                        std::span<float> v) noexcept
{
  const std::size_t a = cfg_.action_dim;
  const std::size_t hd = cfg_.hidden;
  const std::size_t td = cfg_.time_dim;

  std::byte* const mark = scratch_->mark();
  const auto buf = [&](std::size_t n) noexcept {
    return std::span<float>{scratch_->alloc_array<float>(n, kSimdAlign), n};
  };
  std::span<float> h = buf(hd);
  std::span<float> tmp = buf(hd);
  const std::span<float> sinu = buf(td);

  matmul(x, {in_proj_, hd * a}, h, 1uz, a, hd); // in_proj·x

  const std::size_t half = td / 2uz; // sinusoidal time embedding [sin | cos]
  for (std::size_t i{0uz}; i < half; ++i) {
    sinu[i] = std::sin(t * freqs_[i]);
    sinu[half + i] = std::cos(t * freqs_[i]);
  }
  matmul(sinu, {time_proj_, hd * td}, tmp, 1uz, td, hd);

  for (std::size_t i{0uz}; i < hd; ++i)
    h[i] += tmp[i] + c_emb[i]; // fuse x, time, cond
  silu(h);

  for (std::size_t l{0uz}; l < cfg_.mlp_layers; ++l) {
    matmul(h, {layers_[l], hd * hd}, tmp, 1uz, hd, hd);
    silu(tmp);
    std::swap(h, tmp);
  }
  matmul(h, {out_proj_, a * hd}, v, 1uz, hd, a);

  scratch_->reset_to(mark);
}

void FlowHead::sample(std::span<const float> cond, std::span<const float> x0, std::size_t steps,
                      Method method, std::span<float> out) noexcept
{
  const std::size_t a = cfg_.action_dim;
  const std::size_t hd = cfg_.hidden;

  std::byte* const mark = scratch_->mark();
  const auto buf = [&](std::size_t n) noexcept {
    return std::span<float>{scratch_->alloc_array<float>(n, kSimdAlign), n};
  };
  const std::span<float> c_emb = buf(hd);
  matmul(cond, {cond_proj_, hd * cfg_.cond_dim}, c_emb, 1uz, cfg_.cond_dim, hd);

  const std::span<float> x = buf(a);
  const std::span<float> v1 = buf(a);
  const std::span<float> v2 = buf(a);
  const std::span<float> xp = buf(a);
  for (std::size_t i{0uz}; i < a; ++i)
    x[i] = x0[i];

  const float dt = 1.0F / static_cast<float>(steps);
  for (std::size_t k{0uz}; k < steps; ++k) {
    const float t = static_cast<float>(k) * dt;
    velocity(x, t, c_emb, v1);
    if (method == kEuler) {
      for (std::size_t i{0uz}; i < a; ++i)
        x[i] += dt * v1[i];
    } else {
      for (std::size_t i{0uz}; i < a; ++i)
        xp[i] = x[i] + (dt * v1[i]);
      velocity(xp, t + dt, c_emb, v2);
      for (std::size_t i{0uz}; i < a; ++i)
        x[i] += 0.5F * dt * (v1[i] + v2[i]);
    }
  }
  for (std::size_t i{0uz}; i < a; ++i)
    out[i] = x[i];

  scratch_->reset_to(mark);
}

} // namespace fe