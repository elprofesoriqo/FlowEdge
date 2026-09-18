#include "models/mamba/mamba_cuda.h"

#include "kernels/cuda/kernels_cuda_api.h"

namespace fe {

void CudaMambaResident::release() noexcept
{
  using fe::cuda_ops::device_free;
  device_free(norm_f_);
  for (DeviceLayer& layer : layers_) {
    device_free(layer.norm);
    device_free(layer.in_proj);
    device_free(layer.conv_w);
    device_free(layer.conv_b);
    device_free(layer.x_proj);
    device_free(layer.dt_w);
    device_free(layer.dt_b);
    device_free(layer.a_neg);
    device_free(layer.d);
    device_free(layer.out_proj);
    layer = {};
  }
  device_free(hidden_);
  device_free(normed_);
  device_free(xz_);
  device_free(z_);
  device_free(x_cm_);
  device_free(x_conv_);
  device_free(x_sm_);
  device_free(dbl_);
  device_free(dt_in_);
  device_free(dt_);
  device_free(h_);
  device_free(yv_);
  device_free(out_);
  device_free(state_);
  norm_f_ = hidden_ = normed_ = xz_ = z_ = x_cm_ = x_conv_ = x_sm_ = nullptr;
  dbl_ = dt_in_ = dt_ = h_ = yv_ = out_ = state_ = nullptr;
  n_layers_ = d_model_ = d_inner_ = d_state_ = d_conv_ = dt_rank_ = wd_ = max_seq_ = 0;
}

bool CudaMambaResident::upload(const float* host, std::size_t count, float*& dst) noexcept
{
  dst = static_cast<float*>(cuda_ops::device_alloc(count * sizeof(float)));
  return dst != nullptr && host != nullptr &&
         cuda_ops::device_copy_h2d(dst, host, count * sizeof(float));
}

float* CudaMambaResident::device_floats(std::size_t count) noexcept
{
  return static_cast<float*>(cuda_ops::device_alloc(count * sizeof(float)));
}

bool CudaMambaResident::load(std::size_t n_layers, std::size_t d_model, std::size_t d_inner,
                             std::size_t d_state, std::size_t d_conv, std::size_t dt_rank,
                             const float* norm_f, const std::array<LayerHost, kMaxLayers>& layers,
                             std::size_t max_seq) noexcept
{
  release();
  if (!cuda_ops::device_available() || n_layers == 0 || n_layers > kMaxLayers || d_model == 0 ||
      d_inner == 0 || d_state == 0 || d_conv == 0 || dt_rank == 0 || max_seq == 0 ||
      max_seq > kMaxSeq || norm_f == nullptr)
    return false;
  n_layers_ = n_layers;
  d_model_ = d_model;
  d_inner_ = d_inner;
  d_state_ = d_state;
  d_conv_ = d_conv;
  dt_rank_ = dt_rank;
  wd_ = dt_rank + (2 * d_state);
  max_seq_ = max_seq;
  const std::size_t dm = d_model;
  const std::size_t di = d_inner;
  const std::size_t ds = d_state;
  const std::size_t dc = d_conv;
  const std::size_t dr = dt_rank;
  const std::size_t wd = wd_;
  const std::size_t l = max_seq;
  bool ok = upload(norm_f, dm, norm_f_);
  for (std::size_t i = 0; ok && i < n_layers; ++i) {
    const LayerHost& src = layers[i];
    DeviceLayer& dst = layers_[i];
    ok = upload(src.norm, dm, dst.norm) && upload(src.in_proj, 2 * di * dm, dst.in_proj) &&
         upload(src.conv_w, di * dc, dst.conv_w) && upload(src.conv_b, di, dst.conv_b) &&
         upload(src.x_proj, wd * di, dst.x_proj) && upload(src.dt_w, di * dr, dst.dt_w) &&
         upload(src.dt_b, di, dst.dt_b) && upload(src.a_neg, ds * di, dst.a_neg) &&
         upload(src.d, di, dst.d) && upload(src.out_proj, dm * di, dst.out_proj);
  }
  hidden_ = device_floats(l * dm);
  normed_ = device_floats(l * dm);
  xz_ = device_floats(l * 2 * di);
  z_ = device_floats(l * di);
  x_cm_ = device_floats(di * l);
  x_conv_ = device_floats(di * l);
  x_sm_ = device_floats(l * di);
  dbl_ = device_floats(l * wd);
  dt_in_ = device_floats(l * dr);
  dt_ = device_floats(l * di);
  h_ = device_floats(ds * di);
  yv_ = device_floats(l * di);
  out_ = device_floats(l * dm);
  state_ = device_floats(n_layers * di * (dc + ds));
  ok = ok && hidden_ != nullptr && normed_ != nullptr && xz_ != nullptr && z_ != nullptr &&
       x_cm_ != nullptr && x_conv_ != nullptr && x_sm_ != nullptr && dbl_ != nullptr &&
       dt_in_ != nullptr && dt_ != nullptr && h_ != nullptr && yv_ != nullptr && out_ != nullptr &&
       state_ != nullptr;
  if (!ok)
    release();
  return ok;
}

void CudaMambaResident::layer_forward(const DeviceLayer& layer, std::size_t seq_len) noexcept
{
  const std::size_t dm = d_model_;
  const std::size_t di = d_inner_;
  const std::size_t ds = d_state_;
  const std::size_t dr = dt_rank_;
  const std::size_t dc = d_conv_;
  const std::size_t wd = wd_;
  const std::size_t l = seq_len;
  cuda_ops::rmsnorm_device(hidden_, layer.norm, normed_, l, dm);
  cuda_ops::matmul_f32_device(normed_, layer.in_proj, xz_, l, dm, 2 * di);
  cuda_ops::split_xz_device(xz_, x_cm_, z_, l, di);
  cuda_ops::conv1d_causal_device(x_cm_, layer.conv_w, layer.conv_b, x_conv_, di, l, dc);
  cuda_ops::silu_device(x_conv_, di * l);
  cuda_ops::channel_to_seq_device(x_conv_, x_sm_, l, di);
  cuda_ops::matmul_f32_device(x_sm_, layer.x_proj, dbl_, l, di, wd);
  cuda_ops::gather_prefix_device(dbl_, dt_in_, l, wd, dr);
  cuda_ops::matmul_f32_device(dt_in_, layer.dt_w, dt_, l, dr, di);
  cuda_ops::add_bias_rows_device(dt_, layer.dt_b, l, di);
  cuda_ops::softplus_device(dt_, l * di);
  cuda_ops::discretize_and_scan_device(dt_, layer.a_neg, dbl_ + dr, x_sm_, dbl_ + dr + ds, layer.d,
                                       h_, yv_, l, di, ds, true, wd);
  cuda_ops::gate_silu_device(yv_, z_, yv_, l * di);
  cuda_ops::matmul_f32_device(yv_, layer.out_proj, out_, l, di, dm);
  cuda_ops::add_inplace_device(hidden_, out_, l * dm);
}

void CudaMambaResident::decode_layer(const DeviceLayer& layer, float* conv_win, float* h) noexcept
{
  const std::size_t dm = d_model_;
  const std::size_t di = d_inner_;
  const std::size_t ds = d_state_;
  const std::size_t dr = dt_rank_;
  const std::size_t dc = d_conv_;
  const std::size_t wd = wd_;
  cuda_ops::rmsnorm_device(hidden_, layer.norm, normed_, 1, dm);
  cuda_ops::matmul_f32_device(normed_, layer.in_proj, xz_, 1, dm, 2 * di);
  cuda_ops::conv_shift_push_device(conv_win, xz_, z_, di, dc);
  cuda_ops::conv1d_step_device(conv_win, layer.conv_w, layer.conv_b, x_conv_, di, dc);
  cuda_ops::silu_device(x_conv_, di);
  cuda_ops::matmul_f32_device(x_conv_, layer.x_proj, dbl_, 1, di, wd);
  cuda_ops::gather_prefix_device(dbl_, dt_in_, 1, wd, dr);
  cuda_ops::matmul_f32_device(dt_in_, layer.dt_w, dt_, 1, dr, di);
  cuda_ops::add_bias_rows_device(dt_, layer.dt_b, 1, di);
  cuda_ops::softplus_device(dt_, di);
  cuda_ops::discretize_and_scan_device(dt_, layer.a_neg, dbl_ + dr, x_conv_, dbl_ + dr + ds, layer.d,
                                       h, yv_, 1, di, ds, false, wd);
  cuda_ops::gate_silu_device(yv_, z_, yv_, di);
  cuda_ops::matmul_f32_device(yv_, layer.out_proj, out_, 1, di, dm);
  cuda_ops::add_inplace_device(hidden_, out_, dm);
}

bool CudaMambaResident::forward(std::span<const float> input, std::span<float> output,
                                std::size_t seq_len) noexcept
{
  const std::size_t hz = seq_len * d_model_;
  if (seq_len == 0 || seq_len > max_seq_ || input.size() < hz || output.size() < hz)
    return false;
  if (!cuda_ops::device_copy_h2d(hidden_, input.data(), hz * sizeof(float)))
    return false;
  for (std::size_t layer = 0; layer < n_layers_; ++layer)
    layer_forward(layers_[layer], seq_len);
  cuda_ops::rmsnorm_device(hidden_, norm_f_, hidden_, seq_len, d_model_);
  return cuda_ops::device_copy_d2h(output.data(), hidden_, hz * sizeof(float));
}

bool CudaMambaResident::decode(std::span<const float> x, std::span<float> state,
                               std::span<float> out) noexcept
{
  const std::size_t dm = d_model_;
  const std::size_t per = d_inner_ * (d_conv_ + d_state_);
  const std::size_t state_n = n_layers_ * per;
  if (x.size() < dm || out.size() < dm || state.size() < state_n)
    return false;
  if (!cuda_ops::device_copy_h2d(hidden_, x.data(), dm * sizeof(float)) ||
      !cuda_ops::device_copy_h2d(state_, state.data(), state_n * sizeof(float)))
    return false;
  for (std::size_t layer = 0; layer < n_layers_; ++layer) {
    float* slice = state_ + (layer * per);
    decode_layer(layers_[layer], slice, slice + (d_inner_ * d_conv_));
  }
  cuda_ops::rmsnorm_device(hidden_, norm_f_, hidden_, 1, d_model_);
  return cuda_ops::device_copy_d2h(out.data(), hidden_, dm * sizeof(float)) &&
         cuda_ops::device_copy_d2h(state.data(), state_, state_n * sizeof(float));
}

} // namespace fe
