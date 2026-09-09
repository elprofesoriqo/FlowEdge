#include "kernels/kernels.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>

namespace fe {
namespace {

[[nodiscard]] float stable_softplus(float value) noexcept
{
  if (value > 20.0F)
    return value;
  if (value < -20.0F)
    return std::exp(value);
  return std::log1p(std::exp(value));
}

template<std::size_t Kernel, std::size_t Stride>
void conv1d_channels(std::span<const float> x, std::span<const float> weight,
                     std::span<const float> bias, std::span<float> y, std::size_t in_channels,
                     std::size_t out_channels, std::size_t input_length, std::size_t output_length,
                     std::size_t padding, std::size_t first_channel,
                     std::size_t last_channel) noexcept
{
  static_assert(Kernel > 0uz && Stride > 0uz);
  for (std::size_t oc{first_channel}; oc < last_channel; ++oc) {
    float* const output = y.data() + (oc * output_length);
    std::fill_n(output, output_length, bias[oc]);
    for (std::size_t ic{0uz}; ic < in_channels; ++ic) {
      const float* const input = x.data() + (ic * input_length);
      const float* const weights = weight.data() + (((oc * in_channels) + ic) * Kernel);
      for (std::size_t k{0uz}; k < Kernel; ++k) {
        const std::ptrdiff_t offset =
            static_cast<std::ptrdiff_t>(k) - static_cast<std::ptrdiff_t>(padding);
        const std::size_t begin =
            offset < 0 ? (static_cast<std::size_t>(-offset) + Stride - 1uz) / Stride : 0uz;
        const std::ptrdiff_t last_input = static_cast<std::ptrdiff_t>(input_length - 1uz) - offset;
        if (begin >= output_length || last_input < 0)
          continue;
        const std::size_t end =
            std::min(output_length, (static_cast<std::size_t>(last_input) / Stride) + 1uz);
        const float value = weights[k];
        const float* source = input + (begin * Stride) + offset;
        for (std::size_t ot{begin}; ot < end; ++ot, source += Stride)
          output[ot] += *source * value;
      }
    }
  }
}

template<std::size_t Kernel, std::size_t Stride>
void conv_transpose1d_channels(std::span<const float> x, std::span<const float> weight,
                               std::span<const float> bias, std::span<float> y,
                               std::size_t in_channels, std::size_t out_channels,
                               std::size_t input_length, std::size_t output_length,
                               std::size_t padding, std::size_t first_channel,
                               std::size_t last_channel) noexcept
{
  static_assert(Kernel > 0uz && Stride > 0uz);
  for (std::size_t oc{first_channel}; oc < last_channel; ++oc)
    std::fill_n(y.data() + (oc * output_length), output_length, bias[oc]);

  for (std::size_t ic{0uz}; ic < in_channels; ++ic) {
    for (std::size_t it{0uz}; it < input_length; ++it) {
      const float value = x[(ic * input_length) + it];
      const std::size_t origin = it * Stride;
      for (std::size_t oc{first_channel}; oc < last_channel; ++oc) {
        const std::size_t weight_base = ((ic * out_channels) + oc) * Kernel;
        for (std::size_t k{0uz}; k < Kernel; ++k) {
          const std::size_t padded_index = origin + k;
          if (padded_index >= padding) {
            const std::size_t output_index = padded_index - padding;
            if (output_index < output_length)
              y[(oc * output_length) + output_index] += value * weight[weight_base + k];
          }
        }
      }
    }
  }
}

} // namespace

void conv1d(std::span<const float> x, std::span<const float> weight, std::span<const float> bias,
            std::span<float> y, std::size_t in_channels, std::size_t out_channels,
            std::size_t input_length, std::size_t output_length, std::size_t kernel,
            std::size_t stride, std::size_t padding, ThreadPool* pool) noexcept
{
  if (in_channels == 0uz || out_channels == 0uz || input_length == 0uz || output_length == 0uz ||
      kernel == 0uz || stride == 0uz || x.size() < in_channels * input_length ||
      weight.size() < out_channels * in_channels * kernel || bias.size() < out_channels ||
      y.size() < out_channels * output_length) [[unlikely]]
    return;

  const auto operation = [&](std::size_t first, std::size_t last) noexcept {
    if (kernel == 5uz && stride == 1uz)
      conv1d_channels<5uz, 1uz>(x, weight, bias, y, in_channels, out_channels, input_length,
                                output_length, padding, first, last);
    else if (kernel == 3uz && stride == 2uz)
      conv1d_channels<3uz, 2uz>(x, weight, bias, y, in_channels, out_channels, input_length,
                                output_length, padding, first, last);
    else if (kernel == 1uz && stride == 1uz)
      conv1d_channels<1uz, 1uz>(x, weight, bias, y, in_channels, out_channels, input_length,
                                output_length, padding, first, last);
    else {
      for (std::size_t oc{first}; oc < last; ++oc) {
        for (std::size_t ot{0uz}; ot < output_length; ++ot) {
          float sum = bias[oc];
          const std::size_t origin = ot * stride;
          for (std::size_t ic{0uz}; ic < in_channels; ++ic) {
            const std::size_t input_base = ic * input_length;
            const std::size_t weight_base = ((oc * in_channels) + ic) * kernel;
            for (std::size_t k{0uz}; k < kernel; ++k) {
              const std::size_t padded_index = origin + k;
              if (padded_index >= padding) {
                const std::size_t input_index = padded_index - padding;
                if (input_index < input_length)
                  sum += x[input_base + input_index] * weight[weight_base + k];
              }
            }
          }
          y[(oc * output_length) + ot] = sum;
        }
      }
    }
  };
  const unsigned tasks = matmul_task_count(pool, output_length, in_channels * kernel, out_channels,
                                           MatmulWeightType::kF32);
  if (pool != nullptr && tasks > 1u)
    parallel_for(*pool, out_channels, tasks, operation);
  else
    operation(0uz, out_channels);
}

void conv_transpose1d(std::span<const float> x, std::span<const float> weight,
                      std::span<const float> bias, std::span<float> y, std::size_t in_channels,
                      std::size_t out_channels, std::size_t input_length, std::size_t output_length,
                      std::size_t kernel, std::size_t stride, std::size_t padding,
                      ThreadPool* pool) noexcept
{
  if (in_channels == 0uz || out_channels == 0uz || input_length == 0uz || output_length == 0uz ||
      kernel == 0uz || stride == 0uz || x.size() < in_channels * input_length ||
      weight.size() < in_channels * out_channels * kernel || bias.size() < out_channels ||
      y.size() < out_channels * output_length) [[unlikely]]
    return;

  const auto operation = [&](std::size_t first, std::size_t last) noexcept {
    if (kernel == 4uz && stride == 2uz)
      conv_transpose1d_channels<4uz, 2uz>(x, weight, bias, y, in_channels, out_channels,
                                          input_length, output_length, padding, first, last);
    else {
      for (std::size_t oc{first}; oc < last; ++oc)
        std::fill_n(y.data() + (oc * output_length), output_length, bias[oc]);
      for (std::size_t ic{0uz}; ic < in_channels; ++ic) {
        for (std::size_t it{0uz}; it < input_length; ++it) {
          const float value = x[(ic * input_length) + it];
          const std::size_t origin = it * stride;
          for (std::size_t oc{first}; oc < last; ++oc) {
            const std::size_t weight_base = ((ic * out_channels) + oc) * kernel;
            for (std::size_t k{0uz}; k < kernel; ++k) {
              const std::size_t padded_index = origin + k;
              if (padded_index >= padding) {
                const std::size_t output_index = padded_index - padding;
                if (output_index < output_length)
                  y[(oc * output_length) + output_index] += value * weight[weight_base + k];
              }
            }
          }
        }
      }
    }
  };
  const unsigned tasks = matmul_task_count(pool, input_length, in_channels * kernel, out_channels,
                                           MatmulWeightType::kF32);
  if (pool != nullptr && tasks > 1u)
    parallel_for(*pool, out_channels, tasks, operation);
  else
    operation(0uz, out_channels);
}

void group_norm(std::span<float> x, std::span<const float> weight, std::span<const float> bias,
                std::size_t channels, std::size_t length, std::size_t groups,
                float epsilon) noexcept
{
  if (channels == 0uz || length == 0uz || groups == 0uz || channels % groups != 0uz ||
      x.size() < channels * length || weight.size() < channels || bias.size() < channels)
      [[unlikely]]
    return;

  const std::size_t channels_per_group = channels / groups;
  const std::size_t group_values = channels_per_group * length;
  for (std::size_t group{0uz}; group < groups; ++group) {
    const std::size_t first_channel = group * channels_per_group;
    double sum{0.0};
    double square_sum{0.0};
    for (std::size_t local{0uz}; local < channels_per_group; ++local) {
      const std::size_t base = (first_channel + local) * length;
      for (std::size_t t{0uz}; t < length; ++t) {
        const double value = static_cast<double>(x[base + t]);
        sum += value;
        square_sum += value * value;
      }
    }
    const double count = static_cast<double>(group_values);
    const double mean = sum / count;
    const double variance = std::max(0.0, (square_sum / count) - (mean * mean));
    const float inverse_stddev =
        static_cast<float>(1.0 / std::sqrt(variance + static_cast<double>(epsilon)));
    const float mean_f = static_cast<float>(mean);
    for (std::size_t local{0uz}; local < channels_per_group; ++local) {
      const std::size_t channel = first_channel + local;
      const std::size_t base = channel * length;
      for (std::size_t t{0uz}; t < length; ++t)
        x[base + t] = ((x[base + t] - mean_f) * inverse_stddev * weight[channel]) + bias[channel];
    }
  }
}

void mish(std::span<float> x) noexcept
{
  for (float& value : x)
    value *= std::tanh(stable_softplus(value));
}

void film(std::span<float> x, std::span<const float> scale, std::span<const float> bias,
          std::size_t channels, std::size_t length) noexcept
{
  if (x.size() < channels * length || scale.size() < channels || bias.size() < channels)
      [[unlikely]]
    return;
  for (std::size_t channel{0uz}; channel < channels; ++channel) {
    const std::size_t base = channel * length;
    for (std::size_t t{0uz}; t < length; ++t)
      x[base + t] = (scale[channel] * x[base + t]) + bias[channel];
  }
}

void diffusion_timestep_embedding(float timestep, std::span<float> out) noexcept
{
  if (out.empty() || out.size() % 2uz != 0uz) [[unlikely]]
    return;
  const std::size_t half = out.size() / 2uz;
  if (half == 1uz) {
    out[0] = std::sin(timestep);
    out[1] = std::cos(timestep);
    return;
  }
  const float factor = std::log(10000.0F) / static_cast<float>(half - 1uz);
  for (std::size_t index{0uz}; index < half; ++index) {
    const float frequency = std::exp(-factor * static_cast<float>(index));
    const float phase = timestep * frequency;
    out[index] = std::sin(phase);
    out[half + index] = std::cos(phase);
  }
}

} // namespace fe
