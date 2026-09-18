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

} // namespace fe::cuda_ops
