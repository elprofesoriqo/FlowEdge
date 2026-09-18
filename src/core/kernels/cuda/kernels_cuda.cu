#include "kernels/cuda/kernels_cuda_api.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace fe::cuda_ops {
namespace {

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

} // namespace fe::cuda_ops
