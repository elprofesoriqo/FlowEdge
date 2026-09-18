#include "kernels/cuda/kernels_cuda_api.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace fe::cuda_ops {
namespace {

std::atomic<std::uint64_t> g_mallocs{0};

struct GrowBuffer
{
  void* ptr{};
  std::size_t cap{};

  void* ensure(std::size_t bytes) noexcept
  {
    if (bytes <= cap)
      return ptr;
    void* next = nullptr;
    if (cudaMalloc(&next, bytes) != cudaSuccess)
      return nullptr;
    g_mallocs.fetch_add(1, std::memory_order_relaxed);
    if (ptr != nullptr)
      cudaFree(ptr);
    ptr = next;
    cap = bytes;
    return ptr;
  }
};

thread_local GrowBuffer g_a;
thread_local GrowBuffer g_b;
thread_local GrowBuffer g_c;
thread_local GrowBuffer g_d;

[[nodiscard]] bool ready() noexcept
{
  static int cached = -1;
  if (cached < 0) {
    int count = 0;
    cached = (cudaGetDeviceCount(&count) == cudaSuccess && count > 0) ? 1 : 0;
  }
  return cached == 1;
}

bool upload(GrowBuffer& buf, const void* src, std::size_t bytes) noexcept
{
  void* dst = buf.ensure(bytes);
  return dst != nullptr && cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

bool download(void* dst, const void* src, std::size_t bytes) noexcept
{
  return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
}

int grid(int n)
{
  return (n + 255) / 256;
}

__global__ void silu_kernel(float* x, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const float v = x[i];
  x[i] = v / (1.0f + expf(-v));
}

__global__ void mish_kernel(float* x, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const float v = x[i];
  x[i] = v * tanhf(fmaxf(v, 0.0f) + log1pf(expf(-fabsf(v))));
}

__global__ void softplus_kernel(float* x, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const float v = x[i];
  x[i] = fmaxf(v, 0.0f) + log1pf(expf(-fabsf(v)));
}

__global__ void gate_silu_kernel(const float* a, const float* g, float* out, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const float v = g[i];
  out[i] = a[i] * (v / (1.0f + expf(-v)));
}

__global__ void matmul_f32_kernel(const float* in, const float* w, float* out, int rows, int in_dim,
                                  int out_dim)
{
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (o >= out_dim || r >= rows)
    return;
  float acc = 0.0f;
  const float* wr = w + (static_cast<std::size_t>(o) * static_cast<std::size_t>(in_dim));
  const float* ir = in + (static_cast<std::size_t>(r) * static_cast<std::size_t>(in_dim));
  for (int k = 0; k < in_dim; ++k)
    acc += ir[k] * wr[k];
  out[(static_cast<std::size_t>(r) * static_cast<std::size_t>(out_dim)) + o] = acc;
}

__global__ void matmul_bf16_kernel(const float* in, const std::uint16_t* w, float* out, int rows,
                                   int in_dim, int out_dim)
{
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (o >= out_dim || r >= rows)
    return;
  float acc = 0.0f;
  const std::uint16_t* wr = w + (static_cast<std::size_t>(o) * static_cast<std::size_t>(in_dim));
  const float* ir = in + (static_cast<std::size_t>(r) * static_cast<std::size_t>(in_dim));
  for (int k = 0; k < in_dim; ++k) {
    const unsigned bits = static_cast<unsigned>(wr[k]) << 16u;
    acc += ir[k] * __int_as_float(static_cast<int>(bits));
  }
  out[(static_cast<std::size_t>(r) * static_cast<std::size_t>(out_dim)) + o] = acc;
}

__global__ void rmsnorm_kernel(const float* in, const float* weight, float* out, int rows, int dim)
{
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows)
    return;
  const float* ir = in + (static_cast<std::size_t>(r) * static_cast<std::size_t>(dim));
  float* orow = out + (static_cast<std::size_t>(r) * static_cast<std::size_t>(dim));
  float ss = 0.0f;
  for (int i = 0; i < dim; ++i)
    ss += ir[i] * ir[i];
  const float scale = 1.0f / sqrtf((ss / static_cast<float>(dim)) + 1.0e-5f);
  for (int i = 0; i < dim; ++i)
    orow[i] = ir[i] * scale * weight[i];
}

__global__ void conv1d_causal_kernel(const float* x, const float* weight, const float* bias,
                                     float* y, int channels, int length, int kernel)
{
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels)
    return;
  const float* xc = x + (static_cast<std::size_t>(c) * static_cast<std::size_t>(length));
  const float* wc = weight + (static_cast<std::size_t>(c) * static_cast<std::size_t>(kernel));
  float* yc = y + (static_cast<std::size_t>(c) * static_cast<std::size_t>(length));
  const float bc = bias[c];
  for (int t = 0; t < length; ++t)
    yc[t] = bc;
  for (int k = 0; k < kernel; ++k) {
    const float wk = wc[k];
    const int start = (kernel - 1) - k;
    for (int t = start; t < length; ++t)
      yc[t] += wk * xc[t - start];
  }
}

__global__ void conv1d_step_kernel(const float* window, const float* weight, const float* bias,
                                   float* y, int channels, int kernel)
{
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels)
    return;
  const float* wc = window + (static_cast<std::size_t>(c) * static_cast<std::size_t>(kernel));
  const float* kw = weight + (static_cast<std::size_t>(c) * static_cast<std::size_t>(kernel));
  float acc = bias[c];
  for (int k = 0; k < kernel; ++k)
    acc += kw[k] * wc[k];
  y[c] = acc;
}

__global__ void add3_kernel(float* x, const float* y, const float* z, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] += y[i] + z[i];
}

__global__ void add_scaled_kernel(float* x, const float* dx, float scale, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] += scale * dx[i];
}

__global__ void scaled_sum_kernel(const float* x, const float* dx, float scale, float* out, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = x[i] + (scale * dx[i]);
}

__global__ void add_heun_kernel(float* x, const float* k1, const float* k2, float dt, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] += 0.5f * dt * (k1[i] + k2[i]);
}

__global__ void add_rk4_kernel(float* x, const float* k1, const float* k2, const float* k3,
                               const float* k4, float dt, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] += (dt / 6.0f) * (k1[i] + (2.0f * k2[i]) + (2.0f * k3[i]) + k4[i]);
}

__global__ void time_embed_kernel(float t, const float* freqs, float* sinu, int half)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= half)
    return;
  sinu[i] = sinf(t * freqs[i]);
  sinu[half + i] = cosf(t * freqs[i]);
}

__global__ void conv1d_dense_kernel(const float* x, const float* weight, const float* bias, float* y,
                                    int in_channels, int out_channels, int input_length,
                                    int output_length, int kernel, int stride, int padding)
{
  const int ot = blockIdx.x * blockDim.x + threadIdx.x;
  const int oc = blockIdx.y * blockDim.y + threadIdx.y;
  if (ot >= output_length || oc >= out_channels)
    return;
  float sum = bias[oc];
  const int origin = ot * stride;
  for (int ic = 0; ic < in_channels; ++ic) {
    const float* input = x + (static_cast<std::size_t>(ic) * static_cast<std::size_t>(input_length));
    const float* wr = weight + (static_cast<std::size_t>((oc * in_channels) + ic) *
                                static_cast<std::size_t>(kernel));
    for (int k = 0; k < kernel; ++k) {
      const int padded = origin + k;
      if (padded >= padding) {
        const int index = padded - padding;
        if (index < input_length)
          sum += input[index] * wr[k];
      }
    }
  }
  y[(static_cast<std::size_t>(oc) * static_cast<std::size_t>(output_length)) + ot] = sum;
}

__global__ void conv_transpose1d_kernel(const float* x, const float* weight, const float* bias,
                                        float* y, int in_channels, int out_channels,
                                        int input_length, int output_length, int kernel, int stride,
                                        int padding, int k_major)
{
  const int ot = blockIdx.x * blockDim.x + threadIdx.x;
  const int oc = blockIdx.y * blockDim.y + threadIdx.y;
  if (ot >= output_length || oc >= out_channels)
    return;
  float sum = bias[oc];
  for (int ic = 0; ic < in_channels; ++ic) {
    const float* input = x + (static_cast<std::size_t>(ic) * static_cast<std::size_t>(input_length));
    for (int it = 0; it < input_length; ++it) {
      const int origin = it * stride;
      for (int k = 0; k < kernel; ++k) {
        if (origin + k - padding != ot)
          continue;
        const float wv =
            k_major
                ? weight[((((k * out_channels) + oc) * in_channels) + ic)]
                : weight[((((ic * out_channels) + oc) * kernel) + k)];
        sum += input[it] * wv;
      }
    }
  }
  y[(static_cast<std::size_t>(oc) * static_cast<std::size_t>(output_length)) + ot] = sum;
}

__global__ void group_norm_kernel(float* x, const float* weight, const float* bias, int channels,
                                  int length, int groups, float epsilon)
{
  const int group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= groups)
    return;
  const int channels_per_group = channels / groups;
  const int first = group * channels_per_group;
  const int group_values = channels_per_group * length;
  double sum = 0.0;
  double square_sum = 0.0;
  float* const block = x + (static_cast<std::size_t>(first) * static_cast<std::size_t>(length));
  for (int i = 0; i < group_values; ++i) {
    const double value = static_cast<double>(block[i]);
    sum += value;
    square_sum += value * value;
  }
  const double count = static_cast<double>(group_values);
  const double mean = sum / count;
  const double variance = fmax(0.0, (square_sum / count) - (mean * mean));
  const float inv = static_cast<float>(1.0 / sqrt(variance + static_cast<double>(epsilon)));
  const float mean_f = static_cast<float>(mean);
  for (int local = 0; local < channels_per_group; ++local) {
    const int channel = first + local;
    const float scale = inv * weight[channel];
    const float shift = bias[channel];
    float* const row = x + (static_cast<std::size_t>(channel) * static_cast<std::size_t>(length));
    for (int t = 0; t < length; ++t)
      row[t] = ((row[t] - mean_f) * scale) + shift;
  }
}

__global__ void film_kernel(float* x, const float* scale, const float* bias, int channels,
                            int length)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y * blockDim.y + threadIdx.y;
  if (t >= length || c >= channels)
    return;
  const std::size_t i = (static_cast<std::size_t>(c) * static_cast<std::size_t>(length)) + t;
  x[i] = (scale[c] * x[i]) + bias[c];
}

__global__ void diffusion_timestep_kernel(float timestep, float* out, int half, float factor)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= half)
    return;
  const float phase = timestep * expf(-factor * static_cast<float>(i));
  out[i] = sinf(phase);
  out[half + i] = cosf(phase);
}

__global__ void add_bias_kernel(float* x, const float* bias, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  x[i] += bias[i];
}

__global__ void add_inplace_kernel(float* x, const float* y, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] += y[i];
}

__global__ void add_bias_rows_kernel(float* x, const float* bias, int rows, int dim)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = rows * dim;
  if (i >= n)
    return;
  x[i] += bias[i % dim];
}

__global__ void split_xz_kernel(const float* xz, float* x_cm, float* z, int length, int d_inner)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y * blockDim.y + threadIdx.y;
  if (t >= length || c >= d_inner)
    return;
  const std::size_t row = static_cast<std::size_t>(t) * static_cast<std::size_t>(2 * d_inner);
  x_cm[(static_cast<std::size_t>(c) * static_cast<std::size_t>(length)) + t] = xz[row + c];
  z[(static_cast<std::size_t>(t) * static_cast<std::size_t>(d_inner)) + c] =
      xz[row + static_cast<std::size_t>(d_inner) + c];
}

__global__ void channel_to_seq_kernel(const float* x_cm, float* x_sm, int length, int d_inner)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y * blockDim.y + threadIdx.y;
  if (t >= length || c >= d_inner)
    return;
  x_sm[(static_cast<std::size_t>(t) * static_cast<std::size_t>(d_inner)) + c] =
      x_cm[(static_cast<std::size_t>(c) * static_cast<std::size_t>(length)) + t];
}

__global__ void gather_prefix_kernel(const float* rows, float* out, int length, int row_stride,
                                     int width)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int k = blockIdx.y * blockDim.y + threadIdx.y;
  if (t >= length || k >= width)
    return;
  out[(static_cast<std::size_t>(t) * static_cast<std::size_t>(width)) + k] =
      rows[(static_cast<std::size_t>(t) * static_cast<std::size_t>(row_stride)) + k];
}

__global__ void conv_shift_push_kernel(float* window, const float* xz, float* z, int channels,
                                       int kernel)
{
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels)
    return;
  float* w = window + (static_cast<std::size_t>(c) * static_cast<std::size_t>(kernel));
  for (int k = 0; k < kernel - 1; ++k)
    w[k] = w[k + 1];
  w[kernel - 1] = xz[c];
  z[c] = xz[channels + c];
}

__global__ void discretize_and_scan_kernel(const float* delta, const float* a_neg, const float* b,
                                           const float* u, const float* c_proj, const float* d_skip,
                                           float* h, float* y, int length, int d_inner, int d_state,
                                           int row_stride)
{
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= d_inner)
    return;
  for (int t = 0; t < length; ++t) {
    const float dt = delta[(static_cast<std::size_t>(t) * static_cast<std::size_t>(d_inner)) + c];
    const float ut = u[(static_cast<std::size_t>(t) * static_cast<std::size_t>(d_inner)) + c];
    float yt = d_skip[c] * ut;
    const std::size_t brow = static_cast<std::size_t>(t) * static_cast<std::size_t>(row_stride);
    for (int n = 0; n < d_state; ++n) {
      const std::size_t hc = (static_cast<std::size_t>(n) * static_cast<std::size_t>(d_inner)) + c;
      const float da = expf(dt * a_neg[hc]);
      h[hc] = (da * h[hc]) + (dt * b[brow + n] * ut);
      yt += h[hc] * c_proj[brow + n];
    }
    y[(static_cast<std::size_t>(t) * static_cast<std::size_t>(d_inner)) + c] = yt;
  }
}

__global__ void layout_horizon_to_channel_kernel(const float* in, float* out, int horizon,
                                                 int action_dim)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y * blockDim.y + threadIdx.y;
  if (t >= horizon || c >= action_dim)
    return;
  out[(static_cast<std::size_t>(c) * static_cast<std::size_t>(horizon)) + t] =
      in[(static_cast<std::size_t>(t) * static_cast<std::size_t>(action_dim)) + c];
}

__global__ void layout_channel_to_horizon_kernel(const float* in, float* out, int horizon,
                                                 int action_dim)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y * blockDim.y + threadIdx.y;
  if (t >= horizon || c >= action_dim)
    return;
  out[(static_cast<std::size_t>(t) * static_cast<std::size_t>(action_dim)) + c] =
      in[(static_cast<std::size_t>(c) * static_cast<std::size_t>(horizon)) + t];
}

__global__ void ddim_update_kernel(float* x, const float* eps, int n, float sqrt_alpha_t,
                                   float sqrt_beta_t, float sqrt_alpha_prev,
                                   float sqrt_one_minus_prev, int clip, float clip_range)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float predicted_original = (x[i] - (sqrt_beta_t * eps[i])) / sqrt_alpha_t;
  if (clip)
    predicted_original = fminf(clip_range, fmaxf(-clip_range, predicted_original));
  x[i] = (sqrt_alpha_prev * predicted_original) + (sqrt_one_minus_prev * eps[i]);
}

__global__ void ddpm_update_kernel(float* x, const float* eps, const float* noise, int n,
                                   float sqrt_alpha_t, float sqrt_beta_t,
                                   float original_coefficient, float sample_coefficient,
                                   float sqrt_variance, int clip, float clip_range)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float predicted_original = (x[i] - (sqrt_beta_t * eps[i])) / sqrt_alpha_t;
  if (clip)
    predicted_original = fminf(clip_range, fmaxf(-clip_range, predicted_original));
  float next = (original_coefficient * predicted_original) + (sample_coefficient * x[i]);
  if (noise != nullptr)
    next += sqrt_variance * noise[i];
  x[i] = next;
}

} // namespace

bool device_available() noexcept
{
  return ready();
}

void silu(float* x, std::size_t n) noexcept
{
  if (n == 0 || !ready() || !upload(g_a, x, n * sizeof(float)))
    return;
  silu_kernel<<<grid(static_cast<int>(n)), 256>>>(static_cast<float*>(g_a.ptr),
                                                  static_cast<int>(n));
  download(x, g_a.ptr, n * sizeof(float));
}

void mish(float* x, std::size_t n) noexcept
{
  if (n == 0 || !ready() || !upload(g_a, x, n * sizeof(float)))
    return;
  mish_kernel<<<grid(static_cast<int>(n)), 256>>>(static_cast<float*>(g_a.ptr),
                                                  static_cast<int>(n));
  download(x, g_a.ptr, n * sizeof(float));
}

void softplus(float* x, std::size_t n) noexcept
{
  if (n == 0 || !ready() || !upload(g_a, x, n * sizeof(float)))
    return;
  softplus_kernel<<<grid(static_cast<int>(n)), 256>>>(static_cast<float*>(g_a.ptr),
                                                      static_cast<int>(n));
  download(x, g_a.ptr, n * sizeof(float));
}

void gate_silu(const float* a, const float* g, float* out, std::size_t n) noexcept
{
  if (n == 0 || !ready() || !upload(g_a, a, n * sizeof(float)) ||
      !upload(g_b, g, n * sizeof(float)))
    return;
  void* dst = g_c.ensure(n * sizeof(float));
  if (dst == nullptr)
    return;
  gate_silu_kernel<<<grid(static_cast<int>(n)), 256>>>(static_cast<const float*>(g_a.ptr),
                                                       static_cast<const float*>(g_b.ptr),
                                                       static_cast<float*>(dst),
                                                       static_cast<int>(n));
  download(out, dst, n * sizeof(float));
}

void rmsnorm(const float* in, const float* weight, float* out, std::size_t rows,
             std::size_t dim) noexcept
{
  if (rows == 0 || dim == 0 || !ready())
    return;
  const std::size_t in_bytes = rows * dim * sizeof(float);
  if (!upload(g_a, in, in_bytes) || !upload(g_b, weight, dim * sizeof(float)))
    return;
  void* dst = g_c.ensure(in_bytes);
  if (dst == nullptr)
    return;
  rmsnorm_kernel<<<grid(static_cast<int>(rows)), 256>>>(static_cast<const float*>(g_a.ptr),
                                                        static_cast<const float*>(g_b.ptr),
                                                        static_cast<float*>(dst),
                                                        static_cast<int>(rows),
                                                        static_cast<int>(dim));
  download(out, dst, in_bytes);
}

void conv1d_causal(const float* x, const float* weight, const float* bias, float* y,
                   std::size_t channels, std::size_t length, std::size_t kernel) noexcept
{
  if (channels == 0 || length == 0 || kernel == 0 || !ready())
    return;
  if (!upload(g_a, x, channels * length * sizeof(float)) ||
      !upload(g_b, weight, channels * kernel * sizeof(float)) ||
      !upload(g_c, bias, channels * sizeof(float)))
    return;
  void* dst = g_d.ensure(channels * length * sizeof(float));
  if (dst == nullptr)
    return;
  conv1d_causal_kernel<<<grid(static_cast<int>(channels)), 256>>>(
      static_cast<const float*>(g_a.ptr), static_cast<const float*>(g_b.ptr),
      static_cast<const float*>(g_c.ptr), static_cast<float*>(dst), static_cast<int>(channels),
      static_cast<int>(length), static_cast<int>(kernel));
  download(y, dst, channels * length * sizeof(float));
}

void conv1d_step(const float* window, const float* weight, const float* bias, float* y,
                 std::size_t channels, std::size_t kernel) noexcept
{
  if (channels == 0 || kernel == 0 || !ready())
    return;
  if (!upload(g_a, window, channels * kernel * sizeof(float)) ||
      !upload(g_b, weight, channels * kernel * sizeof(float)) ||
      !upload(g_c, bias, channels * sizeof(float)))
    return;
  void* dst = g_d.ensure(channels * sizeof(float));
  if (dst == nullptr)
    return;
  conv1d_step_kernel<<<grid(static_cast<int>(channels)), 256>>>(static_cast<const float*>(g_a.ptr),
                                                                static_cast<const float*>(g_b.ptr),
                                                                static_cast<const float*>(g_c.ptr),
                                                                static_cast<float*>(dst),
                                                                static_cast<int>(channels),
                                                                static_cast<int>(kernel));
  download(y, dst, channels * sizeof(float));
}

void matmul_f32(const float* in, const float* w, float* out, std::size_t rows, std::size_t in_dim,
                std::size_t out_dim) noexcept
{
  if (rows == 0 || in_dim == 0 || out_dim == 0 || !ready())
    return;
  if (!upload(g_a, in, rows * in_dim * sizeof(float)) ||
      !upload(g_b, w, out_dim * in_dim * sizeof(float)))
    return;
  void* dst = g_c.ensure(rows * out_dim * sizeof(float));
  if (dst == nullptr)
    return;
  dim3 block(16, 16);
  dim3 grid_dim(static_cast<unsigned>((out_dim + 15) / 16),
                static_cast<unsigned>((rows + 15) / 16));
  matmul_f32_kernel<<<grid_dim, block>>>(static_cast<const float*>(g_a.ptr),
                                         static_cast<const float*>(g_b.ptr),
                                         static_cast<float*>(dst), static_cast<int>(rows),
                                         static_cast<int>(in_dim), static_cast<int>(out_dim));
  download(out, dst, rows * out_dim * sizeof(float));
}

void matmul_bf16(const float* in, const std::uint16_t* w, float* out, std::size_t rows,
                 std::size_t in_dim, std::size_t out_dim) noexcept
{
  if (rows == 0 || in_dim == 0 || out_dim == 0 || !ready())
    return;
  if (!upload(g_a, in, rows * in_dim * sizeof(float)) ||
      !upload(g_b, w, out_dim * in_dim * sizeof(std::uint16_t)))
    return;
  void* dst = g_c.ensure(rows * out_dim * sizeof(float));
  if (dst == nullptr)
    return;
  dim3 block(16, 16);
  dim3 grid_dim(static_cast<unsigned>((out_dim + 15) / 16),
                static_cast<unsigned>((rows + 15) / 16));
  matmul_bf16_kernel<<<grid_dim, block>>>(static_cast<const float*>(g_a.ptr),
                                          static_cast<const std::uint16_t*>(g_b.ptr),
                                          static_cast<float*>(dst), static_cast<int>(rows),
                                          static_cast<int>(in_dim), static_cast<int>(out_dim));
  download(out, dst, rows * out_dim * sizeof(float));
}

void discretize_and_scan(const float* delta, const float* a_neg, const float* b, const float* u,
                         const float* c_proj, const float* d_skip, float* h, float* y,
                         std::size_t length, std::size_t d_inner, std::size_t d_state,
                         bool reset_state, std::size_t row_stride) noexcept
{
  // Sequential in time. Host reference until a device-resident SSM lands.
  row_stride = row_stride == 0 ? d_state : row_stride;
  if (reset_state)
    std::fill(h, h + (d_inner * d_state), 0.0f);
  for (std::size_t t = 0; t < length; ++t) {
    const float* dt = delta + (t * d_inner);
    const float* ut = u + (t * d_inner);
    const float* bt = b + (t * row_stride);
    const float* ct = c_proj + (t * row_stride);
    float* yt = y + (t * d_inner);
    for (std::size_t c = 0; c < d_inner; ++c)
      yt[c] = d_skip[c] * ut[c];
    for (std::size_t n = 0; n < d_state; ++n) {
      float* hn = h + (n * d_inner);
      const float* an = a_neg + (n * d_inner);
      const float bn = bt[n];
      const float cn = ct[n];
      for (std::size_t c = 0; c < d_inner; ++c) {
        const float da = std::exp(dt[c] * an[c]);
        hn[c] = (da * hn[c]) + (dt[c] * bn * ut[c]);
        yt[c] += hn[c] * cn;
      }
    }
  }
}

std::uint64_t device_malloc_count() noexcept
{
  return g_mallocs.load(std::memory_order_relaxed);
}

void* device_alloc(std::size_t bytes) noexcept
{
  if (bytes == 0 || !ready())
    return nullptr;
  void* ptr = nullptr;
  if (cudaMalloc(&ptr, bytes) != cudaSuccess)
    return nullptr;
  g_mallocs.fetch_add(1, std::memory_order_relaxed);
  return ptr;
}

void device_free(void* ptr) noexcept
{
  if (ptr != nullptr)
    cudaFree(ptr);
}

bool device_copy_h2d(void* dst, const void* src, std::size_t bytes) noexcept
{
  return dst != nullptr && src != nullptr &&
         cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

bool device_copy_d2h(void* dst, const void* src, std::size_t bytes) noexcept
{
  return dst != nullptr && src != nullptr &&
         cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
}

void silu_device(float* x, std::size_t n) noexcept
{
  if (n == 0 || x == nullptr)
    return;
  silu_kernel<<<grid(static_cast<int>(n)), 256>>>(x, static_cast<int>(n));
}

void matmul_f32_device(const float* in, const float* w, float* out, std::size_t rows,
                       std::size_t in_dim, std::size_t out_dim) noexcept
{
  if (rows == 0 || in_dim == 0 || out_dim == 0 || in == nullptr || w == nullptr || out == nullptr)
    return;
  dim3 block(16, 16);
  dim3 grid_dim(static_cast<unsigned>((out_dim + 15) / 16),
                static_cast<unsigned>((rows + 15) / 16));
  matmul_f32_kernel<<<grid_dim, block>>>(in, w, out, static_cast<int>(rows),
                                         static_cast<int>(in_dim), static_cast<int>(out_dim));
}

void add3_device(float* x, const float* y, const float* z, std::size_t n) noexcept
{
  if (n == 0)
    return;
  add3_kernel<<<grid(static_cast<int>(n)), 256>>>(x, y, z, static_cast<int>(n));
}

void add_scaled_device(float* x, const float* dx, float scale, std::size_t n) noexcept
{
  if (n == 0)
    return;
  add_scaled_kernel<<<grid(static_cast<int>(n)), 256>>>(x, dx, scale, static_cast<int>(n));
}

void scaled_sum_device(const float* x, const float* dx, float scale, float* out,
                       std::size_t n) noexcept
{
  if (n == 0)
    return;
  scaled_sum_kernel<<<grid(static_cast<int>(n)), 256>>>(x, dx, scale, out, static_cast<int>(n));
}

void add_heun_device(float* x, const float* k1, const float* k2, float dt, std::size_t n) noexcept
{
  if (n == 0)
    return;
  add_heun_kernel<<<grid(static_cast<int>(n)), 256>>>(x, k1, k2, dt, static_cast<int>(n));
}

void add_rk4_device(float* x, const float* k1, const float* k2, const float* k3, const float* k4,
                    float dt, std::size_t n) noexcept
{
  if (n == 0)
    return;
  add_rk4_kernel<<<grid(static_cast<int>(n)), 256>>>(x, k1, k2, k3, k4, dt, static_cast<int>(n));
}

void time_embed_device(float t, const float* freqs, float* sinu, std::size_t half) noexcept
{
  if (half == 0)
    return;
  time_embed_kernel<<<grid(static_cast<int>(half)), 256>>>(t, freqs, sinu, static_cast<int>(half));
}

void conv1d(const float* x, const float* weight, const float* bias, float* y, std::size_t in_channels,
            std::size_t out_channels, std::size_t input_length, std::size_t output_length,
            std::size_t kernel, std::size_t stride, std::size_t padding) noexcept
{
  if (in_channels == 0 || out_channels == 0 || input_length == 0 || output_length == 0 ||
      kernel == 0 || stride == 0 || !ready())
    return;
  if (!upload(g_a, x, in_channels * input_length * sizeof(float)) ||
      !upload(g_b, weight, out_channels * in_channels * kernel * sizeof(float)) ||
      !upload(g_c, bias, out_channels * sizeof(float)))
    return;
  void* dst = g_d.ensure(out_channels * output_length * sizeof(float));
  if (dst == nullptr)
    return;
  dim3 block(16, 16);
  dim3 grid_dim(static_cast<unsigned>((output_length + 15) / 16),
                static_cast<unsigned>((out_channels + 15) / 16));
  conv1d_dense_kernel<<<grid_dim, block>>>(
      static_cast<const float*>(g_a.ptr), static_cast<const float*>(g_b.ptr),
      static_cast<const float*>(g_c.ptr), static_cast<float*>(dst), static_cast<int>(in_channels),
      static_cast<int>(out_channels), static_cast<int>(input_length),
      static_cast<int>(output_length), static_cast<int>(kernel), static_cast<int>(stride),
      static_cast<int>(padding));
  download(y, dst, out_channels * output_length * sizeof(float));
}

void conv_transpose1d(const float* x, const float* weight, const float* bias, float* y,
                      std::size_t in_channels, std::size_t out_channels, std::size_t input_length,
                      std::size_t output_length, std::size_t kernel, std::size_t stride,
                      std::size_t padding, bool k_major_weights) noexcept
{
  if (in_channels == 0 || out_channels == 0 || input_length == 0 || output_length == 0 ||
      kernel == 0 || stride == 0 || !ready())
    return;
  if (!upload(g_a, x, in_channels * input_length * sizeof(float)) ||
      !upload(g_b, weight, in_channels * out_channels * kernel * sizeof(float)) ||
      !upload(g_c, bias, out_channels * sizeof(float)))
    return;
  void* dst = g_d.ensure(out_channels * output_length * sizeof(float));
  if (dst == nullptr)
    return;
  dim3 block(16, 16);
  dim3 grid_dim(static_cast<unsigned>((output_length + 15) / 16),
                static_cast<unsigned>((out_channels + 15) / 16));
  conv_transpose1d_kernel<<<grid_dim, block>>>(
      static_cast<const float*>(g_a.ptr), static_cast<const float*>(g_b.ptr),
      static_cast<const float*>(g_c.ptr), static_cast<float*>(dst), static_cast<int>(in_channels),
      static_cast<int>(out_channels), static_cast<int>(input_length),
      static_cast<int>(output_length), static_cast<int>(kernel), static_cast<int>(stride),
      static_cast<int>(padding), k_major_weights ? 1 : 0);
  download(y, dst, out_channels * output_length * sizeof(float));
}

void group_norm(float* x, const float* weight, const float* bias, std::size_t channels,
                std::size_t length, std::size_t groups, float epsilon) noexcept
{
  if (channels == 0 || length == 0 || groups == 0 || channels % groups != 0 || !ready())
    return;
  if (!upload(g_a, x, channels * length * sizeof(float)) ||
      !upload(g_b, weight, channels * sizeof(float)) || !upload(g_c, bias, channels * sizeof(float)))
    return;
  group_norm_kernel<<<grid(static_cast<int>(groups)), 256>>>(
      static_cast<float*>(g_a.ptr), static_cast<const float*>(g_b.ptr),
      static_cast<const float*>(g_c.ptr), static_cast<int>(channels), static_cast<int>(length),
      static_cast<int>(groups), epsilon);
  download(x, g_a.ptr, channels * length * sizeof(float));
}

void film(float* x, const float* scale, const float* bias, std::size_t channels,
          std::size_t length) noexcept
{
  if (channels == 0 || length == 0 || !ready())
    return;
  if (!upload(g_a, x, channels * length * sizeof(float)) ||
      !upload(g_b, scale, channels * sizeof(float)) || !upload(g_c, bias, channels * sizeof(float)))
    return;
  dim3 block(16, 16);
  dim3 grid_dim(static_cast<unsigned>((length + 15) / 16),
                static_cast<unsigned>((channels + 15) / 16));
  film_kernel<<<grid_dim, block>>>(static_cast<float*>(g_a.ptr), static_cast<const float*>(g_b.ptr),
                                   static_cast<const float*>(g_c.ptr), static_cast<int>(channels),
                                   static_cast<int>(length));
  download(x, g_a.ptr, channels * length * sizeof(float));
}

void diffusion_timestep_embedding(float timestep, float* out, std::size_t n) noexcept
{
  if (n == 0 || n % 2 != 0 || out == nullptr)
    return;
  const std::size_t half = n / 2;
  if (half == 1) {
    out[0] = std::sin(timestep);
    out[1] = std::cos(timestep);
    return;
  }
  if (!ready())
    return;
  void* dst = g_a.ensure(n * sizeof(float));
  if (dst == nullptr)
    return;
  const float factor = logf(10000.0f) / static_cast<float>(half - 1);
  diffusion_timestep_kernel<<<grid(static_cast<int>(half)), 256>>>(
      timestep, static_cast<float*>(dst), static_cast<int>(half), factor);
  download(out, dst, n * sizeof(float));
}

bool device_copy_d2d(void* dst, const void* src, std::size_t bytes) noexcept
{
  return dst != nullptr && src != nullptr &&
         cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice) == cudaSuccess;
}

void mish_device(float* x, std::size_t n) noexcept
{
  if (n == 0 || x == nullptr)
    return;
  mish_kernel<<<grid(static_cast<int>(n)), 256>>>(x, static_cast<int>(n));
}

void add_bias_device(float* x, const float* bias, std::size_t n) noexcept
{
  if (n == 0 || x == nullptr || bias == nullptr)
    return;
  add_bias_kernel<<<grid(static_cast<int>(n)), 256>>>(x, bias, static_cast<int>(n));
}

void conv1d_device(const float* x, const float* weight, const float* bias, float* y,
                   std::size_t in_channels, std::size_t out_channels, std::size_t input_length,
                   std::size_t output_length, std::size_t kernel, std::size_t stride,
                   std::size_t padding) noexcept
{
  if (in_channels == 0 || out_channels == 0 || input_length == 0 || output_length == 0 ||
      kernel == 0 || stride == 0 || x == nullptr || weight == nullptr || bias == nullptr ||
      y == nullptr)
    return;
  dim3 block(16, 16);
  dim3 grid_dim(static_cast<unsigned>((output_length + 15) / 16),
                static_cast<unsigned>((out_channels + 15) / 16));
  conv1d_dense_kernel<<<grid_dim, block>>>(
      x, weight, bias, y, static_cast<int>(in_channels), static_cast<int>(out_channels),
      static_cast<int>(input_length), static_cast<int>(output_length), static_cast<int>(kernel),
      static_cast<int>(stride), static_cast<int>(padding));
}

void conv_transpose1d_device(const float* x, const float* weight, const float* bias, float* y,
                             std::size_t in_channels, std::size_t out_channels,
                             std::size_t input_length, std::size_t output_length, std::size_t kernel,
                             std::size_t stride, std::size_t padding, bool k_major_weights) noexcept
{
  if (in_channels == 0 || out_channels == 0 || input_length == 0 || output_length == 0 ||
      kernel == 0 || stride == 0 || x == nullptr || weight == nullptr || bias == nullptr ||
      y == nullptr)
    return;
  dim3 block(16, 16);
  dim3 grid_dim(static_cast<unsigned>((output_length + 15) / 16),
                static_cast<unsigned>((out_channels + 15) / 16));
  conv_transpose1d_kernel<<<grid_dim, block>>>(
      x, weight, bias, y, static_cast<int>(in_channels), static_cast<int>(out_channels),
      static_cast<int>(input_length), static_cast<int>(output_length), static_cast<int>(kernel),
      static_cast<int>(stride), static_cast<int>(padding), k_major_weights ? 1 : 0);
}

void group_norm_device(float* x, const float* weight, const float* bias, std::size_t channels,
                       std::size_t length, std::size_t groups, float epsilon) noexcept
{
  if (channels == 0 || length == 0 || groups == 0 || channels % groups != 0 || x == nullptr ||
      weight == nullptr || bias == nullptr)
    return;
  group_norm_kernel<<<grid(static_cast<int>(groups)), 256>>>(
      x, weight, bias, static_cast<int>(channels), static_cast<int>(length),
      static_cast<int>(groups), epsilon);
}

void film_device(float* x, const float* scale, const float* bias, std::size_t channels,
                 std::size_t length) noexcept
{
  if (channels == 0 || length == 0 || x == nullptr || scale == nullptr || bias == nullptr)
    return;
  dim3 block(16, 16);
  dim3 grid_dim(static_cast<unsigned>((length + 15) / 16),
                static_cast<unsigned>((channels + 15) / 16));
  film_kernel<<<grid_dim, block>>>(x, scale, bias, static_cast<int>(channels),
                                   static_cast<int>(length));
}

void diffusion_timestep_device(float timestep, float* out, std::size_t n) noexcept
{
  if (n == 0 || n % 2 != 0 || out == nullptr)
    return;
  const std::size_t half = n / 2;
  if (half == 1) {
    const float host[2] = {sinf(timestep), cosf(timestep)};
    static_cast<void>(device_copy_h2d(out, host, 2 * sizeof(float)));
    return;
  }
  const float factor = logf(10000.0f) / static_cast<float>(half - 1);
  diffusion_timestep_kernel<<<grid(static_cast<int>(half)), 256>>>(
      timestep, out, static_cast<int>(half), factor);
}

void layout_horizon_to_channel_device(const float* in, float* out, std::size_t horizon,
                                      std::size_t action_dim) noexcept
{
  if (horizon == 0 || action_dim == 0 || in == nullptr || out == nullptr)
    return;
  dim3 block(16, 16);
  dim3 grid_dim(static_cast<unsigned>((horizon + 15) / 16),
                static_cast<unsigned>((action_dim + 15) / 16));
  layout_horizon_to_channel_kernel<<<grid_dim, block>>>(in, out, static_cast<int>(horizon),
                                                        static_cast<int>(action_dim));
}

void layout_channel_to_horizon_device(const float* in, float* out, std::size_t horizon,
                                      std::size_t action_dim) noexcept
{
  if (horizon == 0 || action_dim == 0 || in == nullptr || out == nullptr)
    return;
  dim3 block(16, 16);
  dim3 grid_dim(static_cast<unsigned>((horizon + 15) / 16),
                static_cast<unsigned>((action_dim + 15) / 16));
  layout_channel_to_horizon_kernel<<<grid_dim, block>>>(in, out, static_cast<int>(horizon),
                                                        static_cast<int>(action_dim));
}

void ddim_update_device(float* x, const float* eps, std::size_t n, float sqrt_alpha_t,
                        float sqrt_beta_t, float sqrt_alpha_prev, float sqrt_one_minus_prev,
                        bool clip, float clip_range) noexcept
{
  if (n == 0 || x == nullptr || eps == nullptr)
    return;
  ddim_update_kernel<<<grid(static_cast<int>(n)), 256>>>(
      x, eps, static_cast<int>(n), sqrt_alpha_t, sqrt_beta_t, sqrt_alpha_prev, sqrt_one_minus_prev,
      clip ? 1 : 0, clip_range);
}

void ddpm_update_device(float* x, const float* eps, const float* noise, std::size_t n,
                        float sqrt_alpha_t, float sqrt_beta_t, float original_coefficient,
                        float sample_coefficient, float sqrt_variance, bool clip,
                        float clip_range) noexcept
{
  if (n == 0 || x == nullptr || eps == nullptr)
    return;
  ddpm_update_kernel<<<grid(static_cast<int>(n)), 256>>>(
      x, eps, noise, static_cast<int>(n), sqrt_alpha_t, sqrt_beta_t, original_coefficient,
      sample_coefficient, sqrt_variance, clip ? 1 : 0, clip_range);
}

void rmsnorm_device(const float* in, const float* weight, float* out, std::size_t rows,
                    std::size_t dim) noexcept
{
  if (rows == 0 || dim == 0 || in == nullptr || weight == nullptr || out == nullptr)
    return;
  rmsnorm_kernel<<<grid(static_cast<int>(rows)), 256>>>(in, weight, out, static_cast<int>(rows),
                                                        static_cast<int>(dim));
}

void conv1d_causal_device(const float* x, const float* weight, const float* bias, float* y,
                          std::size_t channels, std::size_t length, std::size_t kernel) noexcept
{
  if (channels == 0 || length == 0 || kernel == 0 || x == nullptr || weight == nullptr ||
      bias == nullptr || y == nullptr)
    return;
  conv1d_causal_kernel<<<grid(static_cast<int>(channels)), 256>>>(
      x, weight, bias, y, static_cast<int>(channels), static_cast<int>(length),
      static_cast<int>(kernel));
}

void conv1d_step_device(const float* window, const float* weight, const float* bias, float* y,
                        std::size_t channels, std::size_t kernel) noexcept
{
  if (channels == 0 || kernel == 0 || window == nullptr || weight == nullptr || bias == nullptr ||
      y == nullptr)
    return;
  conv1d_step_kernel<<<grid(static_cast<int>(channels)), 256>>>(
      window, weight, bias, y, static_cast<int>(channels), static_cast<int>(kernel));
}

void softplus_device(float* x, std::size_t n) noexcept
{
  if (n == 0 || x == nullptr)
    return;
  softplus_kernel<<<grid(static_cast<int>(n)), 256>>>(x, static_cast<int>(n));
}

void gate_silu_device(const float* a, const float* g, float* out, std::size_t n) noexcept
{
  if (n == 0 || a == nullptr || g == nullptr || out == nullptr)
    return;
  gate_silu_kernel<<<grid(static_cast<int>(n)), 256>>>(a, g, out, static_cast<int>(n));
}

void add_inplace_device(float* x, const float* y, std::size_t n) noexcept
{
  if (n == 0 || x == nullptr || y == nullptr)
    return;
  add_inplace_kernel<<<grid(static_cast<int>(n)), 256>>>(x, y, static_cast<int>(n));
}

void add_bias_rows_device(float* x, const float* bias, std::size_t rows, std::size_t dim) noexcept
{
  if (rows == 0 || dim == 0 || x == nullptr || bias == nullptr)
    return;
  add_bias_rows_kernel<<<grid(static_cast<int>(rows * dim)), 256>>>(x, bias, static_cast<int>(rows),
                                                                    static_cast<int>(dim));
}

void split_xz_device(const float* xz, float* x_cm, float* z, std::size_t length,
                     std::size_t d_inner) noexcept
{
  if (length == 0 || d_inner == 0 || xz == nullptr || x_cm == nullptr || z == nullptr)
    return;
  dim3 block(16, 16);
  dim3 grid_dim(static_cast<unsigned>((length + 15) / 16),
                static_cast<unsigned>((d_inner + 15) / 16));
  split_xz_kernel<<<grid_dim, block>>>(xz, x_cm, z, static_cast<int>(length),
                                       static_cast<int>(d_inner));
}

void channel_to_seq_device(const float* x_cm, float* x_sm, std::size_t length,
                           std::size_t d_inner) noexcept
{
  if (length == 0 || d_inner == 0 || x_cm == nullptr || x_sm == nullptr)
    return;
  dim3 block(16, 16);
  dim3 grid_dim(static_cast<unsigned>((length + 15) / 16),
                static_cast<unsigned>((d_inner + 15) / 16));
  channel_to_seq_kernel<<<grid_dim, block>>>(x_cm, x_sm, static_cast<int>(length),
                                             static_cast<int>(d_inner));
}

void gather_prefix_device(const float* rows, float* out, std::size_t length, std::size_t row_stride,
                          std::size_t width) noexcept
{
  if (length == 0 || width == 0 || rows == nullptr || out == nullptr)
    return;
  dim3 block(16, 16);
  dim3 grid_dim(static_cast<unsigned>((length + 15) / 16),
                static_cast<unsigned>((width + 15) / 16));
  gather_prefix_kernel<<<grid_dim, block>>>(rows, out, static_cast<int>(length),
                                            static_cast<int>(row_stride), static_cast<int>(width));
}

void conv_shift_push_device(float* window, const float* xz, float* z, std::size_t channels,
                            std::size_t kernel) noexcept
{
  if (channels == 0 || kernel == 0 || window == nullptr || xz == nullptr || z == nullptr)
    return;
  conv_shift_push_kernel<<<grid(static_cast<int>(channels)), 256>>>(
      window, xz, z, static_cast<int>(channels), static_cast<int>(kernel));
}

void discretize_and_scan_device(const float* delta, const float* a_neg, const float* b,
                                const float* u, const float* c_proj, const float* d_skip, float* h,
                                float* y, std::size_t length, std::size_t d_inner,
                                std::size_t d_state, bool reset_state,
                                std::size_t row_stride) noexcept
{
  if (length == 0 || d_inner == 0 || d_state == 0 || delta == nullptr || a_neg == nullptr ||
      b == nullptr || u == nullptr || c_proj == nullptr || d_skip == nullptr || h == nullptr ||
      y == nullptr)
    return;
  row_stride = row_stride == 0 ? d_state : row_stride;
  if (reset_state)
    static_cast<void>(cudaMemset(h, 0, d_inner * d_state * sizeof(float)));
  discretize_and_scan_kernel<<<grid(static_cast<int>(d_inner)), 256>>>(
      delta, a_neg, b, u, c_proj, d_skip, h, y, static_cast<int>(length), static_cast<int>(d_inner),
      static_cast<int>(d_state), static_cast<int>(row_stride));
}

} // namespace fe::cuda_ops
