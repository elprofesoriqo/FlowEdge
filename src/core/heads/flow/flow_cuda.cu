#include "heads/flow/flow_cuda.h"

#include "kernels/cuda/kernels_cuda_api.h"

namespace fe {

CudaFlowResident::~CudaFlowResident()
{
  release();
}

void CudaFlowResident::release() noexcept
{
  using fe::cuda_ops::device_free;
  device_free(in_proj_);
  device_free(time_proj_);
  device_free(cond_proj_);
  device_free(out_proj_);
  for (float* layer : layers_)
    device_free(layer);
  device_free(freqs_);
  device_free(cond_);
  device_free(c_emb_);
  device_free(h_);
  device_free(tmp_);
  device_free(sinu_);
  device_free(x_);
  device_free(k1_);
  device_free(k2_);
  device_free(k3_);
  device_free(k4_);
  device_free(probe_);
  in_proj_ = time_proj_ = cond_proj_ = out_proj_ = freqs_ = nullptr;
  cond_ = c_emb_ = h_ = tmp_ = sinu_ = x_ = k1_ = k2_ = k3_ = k4_ = probe_ = nullptr;
  layers_.fill(nullptr);
}

bool CudaFlowResident::upload(const float* host, std::size_t count, float*& dst) noexcept
{
  dst = static_cast<float*>(cuda_ops::device_alloc(count * sizeof(float)));
  return dst != nullptr && host != nullptr &&
         cuda_ops::device_copy_h2d(dst, host, count * sizeof(float));
}

bool CudaFlowResident::load(std::size_t action_dim, std::size_t cond_dim, std::size_t hidden,
                            std::size_t time_dim, std::size_t mlp_layers, const float* in_proj,
                            const float* time_proj, const float* cond_proj, const float* out_proj,
                            const std::array<const float*, kMaxMlp>& layers,
                            const float* freqs) noexcept
{
  release();
  if (!cuda_ops::device_available() || freqs == nullptr || mlp_layers > kMaxMlp)
    return false;
  action_dim_ = action_dim;
  cond_dim_ = cond_dim;
  hidden_ = hidden;
  time_dim_ = time_dim;
  mlp_layers_ = mlp_layers;
  if (!upload(in_proj, hidden * action_dim, in_proj_) ||
      !upload(time_proj, hidden * time_dim, time_proj_) ||
      !upload(cond_proj, hidden * cond_dim, cond_proj_) ||
      !upload(out_proj, action_dim * hidden, out_proj_))
    return false;
  for (std::size_t i = 0; i < mlp_layers; ++i) {
    if (!upload(layers[i], hidden * hidden, layers_[i]))
      return false;
  }
  if (!upload(freqs, time_dim / 2, freqs_))
    return false;
  cond_ = static_cast<float*>(cuda_ops::device_alloc(cond_dim * sizeof(float)));
  c_emb_ = static_cast<float*>(cuda_ops::device_alloc(hidden * sizeof(float)));
  h_ = static_cast<float*>(cuda_ops::device_alloc(hidden * sizeof(float)));
  tmp_ = static_cast<float*>(cuda_ops::device_alloc(hidden * sizeof(float)));
  sinu_ = static_cast<float*>(cuda_ops::device_alloc(time_dim * sizeof(float)));
  x_ = static_cast<float*>(cuda_ops::device_alloc(action_dim * sizeof(float)));
  k1_ = static_cast<float*>(cuda_ops::device_alloc(action_dim * sizeof(float)));
  k2_ = static_cast<float*>(cuda_ops::device_alloc(action_dim * sizeof(float)));
  k3_ = static_cast<float*>(cuda_ops::device_alloc(action_dim * sizeof(float)));
  k4_ = static_cast<float*>(cuda_ops::device_alloc(action_dim * sizeof(float)));
  probe_ = static_cast<float*>(cuda_ops::device_alloc(action_dim * sizeof(float)));
  return cond_ != nullptr && c_emb_ != nullptr && h_ != nullptr && tmp_ != nullptr &&
         sinu_ != nullptr && x_ != nullptr && k1_ != nullptr && k2_ != nullptr && k3_ != nullptr &&
         k4_ != nullptr && probe_ != nullptr;
}

bool CudaFlowResident::begin(std::span<const float> cond, std::span<const float> x0) noexcept
{
  if (cond.size() != cond_dim_ || x0.size() != action_dim_)
    return false;
  if (!cuda_ops::device_copy_h2d(cond_, cond.data(), cond.size() * sizeof(float)) ||
      !cuda_ops::device_copy_h2d(x_, x0.data(), x0.size() * sizeof(float)))
    return false;
  cuda_ops::matmul_f32_device(cond_, cond_proj_, c_emb_, 1, cond_dim_, hidden_);
  return true;
}

void CudaFlowResident::velocity(float* x, float t, float* v) noexcept
{
  const std::size_t a = action_dim_;
  const std::size_t hd = hidden_;
  const std::size_t td = time_dim_;
  cuda_ops::matmul_f32_device(x, in_proj_, h_, 1, a, hd);
  cuda_ops::time_embed_device(t, freqs_, sinu_, td / 2);
  cuda_ops::matmul_f32_device(sinu_, time_proj_, tmp_, 1, td, hd);
  cuda_ops::add3_device(h_, tmp_, c_emb_, hd);
  cuda_ops::silu_device(h_, hd);
  float* cur = h_;
  float* nxt = tmp_;
  for (std::size_t layer = 0; layer < mlp_layers_; ++layer) {
    cuda_ops::matmul_f32_device(cur, layers_[layer], nxt, 1, hd, hd);
    cuda_ops::silu_device(nxt, hd);
    float* swap = cur;
    cur = nxt;
    nxt = swap;
  }
  cuda_ops::matmul_f32_device(cur, out_proj_, v, 1, hd, a);
}

std::size_t CudaFlowResident::advance(std::size_t step_budget, Method method, std::size_t steps,
                                      std::size_t next_step, std::span<float> out) noexcept
{
  if (out.size() < action_dim_ || steps == 0)
    return 0;
  const std::size_t remaining = next_step < steps ? steps - next_step : 0;
  const std::size_t todo = step_budget < remaining ? step_budget : remaining;
  const float dt = 1.0F / static_cast<float>(steps);
  const std::size_t a = action_dim_;
  std::size_t completed = 0;
  for (std::size_t step = next_step; step < next_step + todo; ++step) {
    const float t = static_cast<float>(step) * dt;
    velocity(x_, t, k1_);
    if (method == kEuler) {
      cuda_ops::add_scaled_device(x_, k1_, dt, a);
    } else if (method == kHeun) {
      cuda_ops::scaled_sum_device(x_, k1_, dt, probe_, a);
      velocity(probe_, t + dt, k2_);
      cuda_ops::add_heun_device(x_, k1_, k2_, dt, a);
    } else {
      const float half_dt = 0.5F * dt;
      cuda_ops::scaled_sum_device(x_, k1_, half_dt, probe_, a);
      velocity(probe_, t + half_dt, k2_);
      cuda_ops::scaled_sum_device(x_, k2_, half_dt, probe_, a);
      velocity(probe_, t + half_dt, k3_);
      cuda_ops::scaled_sum_device(x_, k3_, dt, probe_, a);
      velocity(probe_, t + dt, k4_);
      cuda_ops::add_rk4_device(x_, k1_, k2_, k3_, k4_, dt, a);
    }
    ++completed;
  }
  if (!cuda_ops::device_copy_d2h(out.data(), x_, a * sizeof(float)))
    return 0;
  return completed;
}

} // namespace fe
