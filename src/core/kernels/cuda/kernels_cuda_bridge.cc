#include "kernels/cuda/kernels_cuda_api.h"
#include "kernels/kernels.h"

namespace fe {

void conv1d_causal(std::span<const float> x, std::span<const float> weight,
                   std::span<const float> bias, std::span<float> y, std::size_t channels,
                   std::size_t length, std::size_t kernel) noexcept
{
  cuda_ops::conv1d_causal(x.data(), weight.data(), bias.data(), y.data(), channels, length, kernel);
}

void conv1d_step(std::span<const float> window, std::span<const float> weight,
                 std::span<const float> bias, std::span<float> y, std::size_t channels,
                 std::size_t kernel) noexcept
{
  cuda_ops::conv1d_step(window.data(), weight.data(), bias.data(), y.data(), channels, kernel);
}

void rmsnorm(std::span<const float> in, std::span<const float> weight, std::span<float> out,
             std::size_t rows, std::size_t dim) noexcept
{
  cuda_ops::rmsnorm(in.data(), weight.data(), out.data(), rows, dim);
}

void gate_silu(std::span<const float> a, std::span<const float> g, std::span<float> out) noexcept
{
  cuda_ops::gate_silu(a.data(), g.data(), out.data(), out.size());
}

void silu(std::span<float> x) noexcept
{
  cuda_ops::silu(x.data(), x.size());
}

void mish(std::span<float> x) noexcept
{
  cuda_ops::mish(x.data(), x.size());
}

void softplus(std::span<float> x) noexcept
{
  cuda_ops::softplus(x.data(), x.size());
}

void matmul(std::span<const float> in, std::span<const float> w, std::span<float> out,
            std::size_t rows, std::size_t in_dim, std::size_t out_dim, ThreadPool*) noexcept
{
  cuda_ops::matmul_f32(in.data(), w.data(), out.data(), rows, in_dim, out_dim);
}

void matmul(std::span<const float> in, std::span<const std::uint16_t> w, std::span<float> out,
            std::size_t rows, std::size_t in_dim, std::size_t out_dim, ThreadPool*) noexcept
{
  cuda_ops::matmul_bf16(in.data(), w.data(), out.data(), rows, in_dim, out_dim);
}

void discretize_and_scan(std::span<const float> delta, std::span<const float> a_neg,
                         std::span<const float> b, std::span<const float> u,
                         std::span<const float> c_proj, std::span<const float> d_skip,
                         std::span<float> h, std::span<float> y, std::size_t length,
                         std::size_t d_inner, std::size_t d_state, bool reset_state,
                         std::size_t row_stride) noexcept
{
  cuda_ops::discretize_and_scan(delta.data(), a_neg.data(), b.data(), u.data(), c_proj.data(),
                                d_skip.data(), h.data(), y.data(), length, d_inner, d_state,
                                reset_state, row_stride);
}

void conv1d(std::span<const float> x, std::span<const float> weight, std::span<const float> bias,
            std::span<float> y, std::size_t in_channels, std::size_t out_channels,
            std::size_t input_length, std::size_t output_length, std::size_t kernel,
            std::size_t stride, std::size_t padding, ThreadPool*, std::span<float>) noexcept
{
  if (x.size() < in_channels * input_length ||
      weight.size() < out_channels * in_channels * kernel || bias.size() < out_channels ||
      y.size() < out_channels * output_length)
    return;
  cuda_ops::conv1d(x.data(), weight.data(), bias.data(), y.data(), in_channels, out_channels,
                   input_length, output_length, kernel, stride, padding);
}

void conv_transpose1d(std::span<const float> x, std::span<const float> weight,
                      std::span<const float> bias, std::span<float> y, std::size_t in_channels,
                      std::size_t out_channels, std::size_t input_length, std::size_t output_length,
                      std::size_t kernel, std::size_t stride, std::size_t padding, ThreadPool*,
                      std::span<float>, bool k_major_weights) noexcept
{
  if (x.size() < in_channels * input_length ||
      weight.size() < in_channels * out_channels * kernel || bias.size() < out_channels ||
      y.size() < out_channels * output_length)
    return;
  cuda_ops::conv_transpose1d(x.data(), weight.data(), bias.data(), y.data(), in_channels,
                             out_channels, input_length, output_length, kernel, stride, padding,
                             k_major_weights);
}

void group_norm(std::span<float> x, std::span<const float> weight, std::span<const float> bias,
                std::size_t channels, std::size_t length, std::size_t groups,
                float epsilon) noexcept
{
  if (x.size() < channels * length || weight.size() < channels || bias.size() < channels)
    return;
  cuda_ops::group_norm(x.data(), weight.data(), bias.data(), channels, length, groups, epsilon);
}

void film(std::span<float> x, std::span<const float> scale, std::span<const float> bias,
          std::size_t channels, std::size_t length) noexcept
{
  if (x.size() < channels * length || scale.size() < channels || bias.size() < channels)
    return;
  cuda_ops::film(x.data(), scale.data(), bias.data(), channels, length);
}

void diffusion_timestep_embedding(float timestep, std::span<float> out) noexcept
{
  cuda_ops::diffusion_timestep_embedding(timestep, out.data(), out.size());
}

} // namespace fe
