#include "flow.h"

#include "kernels/kernels.h"
#include "kernels/span_ops.h"
#include "loader/tensor_key.h"
#include "loader/weight_ops.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <string_view>
#include <utility>

namespace fe {
namespace {

[[nodiscard]] bool is_matrix(const TensorView* tensor, std::size_t rows,
                             std::size_t columns) noexcept
{
  return tensor != nullptr && tensor->ndim == 2uz && tensor->shape[0] == rows &&
         tensor->shape[1] == columns && is_matmul_weight(tensor);
}

std::string_view layer_key(std::span<char> buf, std::size_t layer) noexcept
{
  TensorKeyBuilder key{buf};
  if (!key.append("flow.layers.") || !key.append(layer) || !key.append(".weight"))
    return {};
  return key.view();
}

} // namespace

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
  if (cfg_.hidden == 0uz || cfg_.action_dim == 0uz || cfg_.time_dim == 0uz ||
      cfg_.cond_dim == 0uz || cfg_.time_dim % 2uz != 0uz ||
      !is_matrix(in, cfg_.hidden, cfg_.action_dim) || !is_matrix(tp, cfg_.hidden, cfg_.time_dim) ||
      !is_matrix(cp, cfg_.hidden, cfg_.cond_dim) || !is_matrix(op, cfg_.action_dim, cfg_.hidden))
    return;

  in_proj_ = weight_view(in);
  time_proj_ = weight_view(tp);
  cond_proj_ = weight_view(cp);
  out_proj_ = weight_view(op);

  std::size_t n{0uz};
  for (; n < kMaxMlp; ++n) {
    std::array<char, 48> buf{};
    const TensorView* lw = find_tensor(weights, layer_key(buf, n));
    if (lw == nullptr)
      break;
    if (!is_matrix(lw, cfg_.hidden, cfg_.hidden))
      return;
    layers_[n] = weight_view(lw);
  }
  cfg_.mlp_layers = n;

  const std::size_t half = cfg_.time_dim / 2uz; // sinusoidal freqs are constant
  auto* const f = scratch.alloc_array<float, kSimdAlign>(half);
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

  matmul_weight(x, in_proj_, h, 1uz, a, hd, pool_); // in_proj·x

  const std::size_t half = td / 2uz;                // sinusoidal time embedding [sin | cos]
  for (std::size_t i{0uz}; i < half; ++i) {
    sinu[i] = std::sin(t * freqs_[i]);
    sinu[half + i] = std::cos(t * freqs_[i]);
  }
  matmul_weight(sinu, time_proj_, tmp, 1uz, td, hd, pool_);

  add3_inplace(h, tmp, c_emb); // fuse x, time, cond
  silu(h);

  for (std::size_t l{0uz}; l < cfg_.mlp_layers; ++l) {
    matmul_weight(h, layers_[l], tmp, 1uz, hd, hd, pool_);
    silu(tmp);
    std::swap(h, tmp);
  }
  matmul_weight(h, out_proj_, v, 1uz, hd, a, pool_);

  scratch_->reset_to(mark);
}

void FlowHead::sample(std::span<const float> cond, std::span<const float> x0, std::size_t steps,
                      Method method, std::span<float> out) noexcept
{
  const std::size_t a = cfg_.action_dim;
  const std::size_t hd = cfg_.hidden;

  std::byte* const mark = scratch_->mark();
  const std::span<float> c_emb = arena_span(hd);
  matmul_weight(cond, cond_proj_, c_emb, 1uz, cfg_.cond_dim, hd);

  const std::span<float> x = arena_span(a);
  const std::span<float> k1 = arena_span(a);
  const std::span<float> k2 = arena_span(a);
  const std::span<float> k3 = arena_span(a);
  const std::span<float> k4 = arena_span(a);
  const std::span<float> xp = arena_span(a);
  copy_span(x0, x);

  const float dt = 1.0F / static_cast<float>(steps);
  for (std::size_t k{0uz}; k < steps; ++k) {
    const float t = static_cast<float>(k) * dt;
    velocity(x, t, c_emb, k1);
    if (method == kEuler) {
      add_scaled(x, k1, dt);
    } else if (method == kHeun) {
      scaled_sum(x, k1, dt, xp);
      velocity(xp, t + dt, c_emb, k2);
      add_heun(x, k1, k2, dt);
    } else { // kRK4: classic 4-stage
      const float h = 0.5F * dt;
      scaled_sum(x, k1, h, xp);
      velocity(xp, t + h, c_emb, k2);
      scaled_sum(x, k2, h, xp);
      velocity(xp, t + h, c_emb, k3);
      scaled_sum(x, k3, dt, xp);
      velocity(xp, t + dt, c_emb, k4);
      add_rk4(x, k1, k2, k3, k4, dt);
    }
  }
  copy_span(x, out);

  scratch_->reset_to(mark);
}

} // namespace fe
