#include "flow.h"

#include "kernels/kernels.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <string_view>
#include <utility>

namespace fe {
FlowHead::FlowHead(std::span<const TensorView> weights, Arena& scratch) noexcept
    : scratch_{&scratch}
{
  const TensorView* in = find_tensor(weights, "flow.in_proj.weight");
  const TensorView* tp = find_tensor(weights, "flow.time_proj.weight");
  const TensorView* cp = find_tensor(weights, "flow.cond_proj.weight");
  const TensorView* op = find_tensor(weights, "flow.out_proj.weight");
  if ((in == nullptr) || (tp == nullptr) || (cp == nullptr) || (op == nullptr))
    return;

  cfg_.hidden = in->shape[0];
  cfg_.action_dim = in->shape[1];
  cfg_.time_dim = tp->shape[1];
  cfg_.cond_dim = cp->shape[1];
  in_proj_ = in->as_bf16();
  time_proj_ = tp->as_bf16();
  cond_proj_ = cp->as_bf16();
  out_proj_ = op->as_bf16();

  std::size_t n{0uz};
  for (; n < kMaxMlp; ++n) {
    std::array<char, 48> buf{};
    std::size_t p{0uz};
    for (const char ch : std::string_view{"flow.layers."})
      buf[p++] = ch;
    // multi-digit: safe when kMaxMlp is raised past 9
    std::array<char, 4> digs{};
    std::size_t nd{0uz};
    for (std::size_t v = n; v != 0uz; v /= 10uz)
      digs[nd++] = static_cast<char>('0' + (v % 10uz));
    if (nd == 0uz)
      digs[nd++] = '0';
    while (nd > 0uz)
      buf[p++] = digs[--nd];
    for (const char ch : std::string_view{".weight"})
      buf[p++] = ch;
    const TensorView* lw = find_tensor(weights, {buf.data(), p});
    if (lw == nullptr)
      break;
    layers_[n] = lw->as_bf16();
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
  std::span<float> h = arena_span(hd);
  std::span<float> tmp = arena_span(hd);
  const std::span<float> sinu = arena_span(td);

  matmul(x, {in_proj_, hd * a}, h, 1uz, a, hd); // in_proj·x

  const std::size_t half = td / 2uz;            // sinusoidal time embedding [sin | cos]
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
  const std::span<float> c_emb = arena_span(hd);
  matmul(cond, {cond_proj_, hd * cfg_.cond_dim}, c_emb, 1uz, cfg_.cond_dim, hd);

  const std::span<float> x = arena_span(a);
  const std::span<float> k1 = arena_span(a);
  const std::span<float> k2 = arena_span(a);
  const std::span<float> k3 = arena_span(a);
  const std::span<float> k4 = arena_span(a);
  const std::span<float> xp = arena_span(a);
  for (std::size_t i{0uz}; i < a; ++i)
    x[i] = x0[i];

  const float dt = 1.0F / static_cast<float>(steps);
  for (std::size_t k{0uz}; k < steps; ++k) {
    const float t = static_cast<float>(k) * dt;
    velocity(x, t, c_emb, k1);
    if (method == kEuler) {
      for (std::size_t i{0uz}; i < a; ++i)
        x[i] += dt * k1[i];
    } else if (method == kHeun) {
      for (std::size_t i{0uz}; i < a; ++i)
        xp[i] = x[i] + (dt * k1[i]);
      velocity(xp, t + dt, c_emb, k2);
      for (std::size_t i{0uz}; i < a; ++i)
        x[i] += 0.5F * dt * (k1[i] + k2[i]);
    } else { // kRK4: classic 4-stage
      const float h = 0.5F * dt;
      for (std::size_t i{0uz}; i < a; ++i)
        xp[i] = x[i] + (h * k1[i]);
      velocity(xp, t + h, c_emb, k2);
      for (std::size_t i{0uz}; i < a; ++i)
        xp[i] = x[i] + (h * k2[i]);
      velocity(xp, t + h, c_emb, k3);
      for (std::size_t i{0uz}; i < a; ++i)
        xp[i] = x[i] + (dt * k3[i]);
      velocity(xp, t + dt, c_emb, k4);
      for (std::size_t i{0uz}; i < a; ++i)
        x[i] += (dt / 6.0F) * (k1[i] + (2.0F * k2[i]) + (2.0F * k3[i]) + k4[i]);
    }
  }
  for (std::size_t i{0uz}; i < a; ++i)
    out[i] = x[i];

  scratch_->reset_to(mark);
}

} // namespace fe