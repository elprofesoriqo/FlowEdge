#pragma once

#include "kernels/kernels.h"
#include "loader/safetensors.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace fe {

inline bool is_matmul_weight(const TensorView* t) noexcept
{
  return (t != nullptr) && (t->is_f32() || t->is_bf16());
}

inline WeightView weight_view(const TensorView* t) noexcept
{
  return (t != nullptr) ? WeightView{t->data, t->dtype} : WeightView{};
}

inline void matmul_weight(std::span<const float> in, WeightView w, std::span<float> out,
                          std::size_t rows, std::size_t in_dim, std::size_t out_dim,
                          ThreadPool* pool = nullptr) noexcept
{
  if (w.dtype == TensorView::Dtype::BF16) {
    matmul(in, {static_cast<const std::uint16_t*>(w.data), in_dim * out_dim}, out, rows, in_dim,
           out_dim, pool);
    return;
  }
  matmul(in, {static_cast<const float*>(w.data), in_dim * out_dim}, out, rows, in_dim, out_dim,
         pool);
}

} // namespace fe
