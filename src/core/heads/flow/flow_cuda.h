#pragma once

#include <array>
#include <cstddef>
#include <span>

namespace fe {

// Persistent device copies of flow.* weights and ODE scratch. Load allocates;
// sample/advance must not cudaMalloc.
class CudaFlowResident
{
public:
  static constexpr std::size_t kMaxMlp = 8;

  enum Method : int
  {
    kEuler = 0,
    kHeun = 1,
    kRK4 = 2
  };

  CudaFlowResident() = default;
  ~CudaFlowResident();
  CudaFlowResident(const CudaFlowResident&) = delete;
  CudaFlowResident(CudaFlowResident&&) = delete;
  CudaFlowResident& operator=(const CudaFlowResident&) = delete;
  CudaFlowResident& operator=(CudaFlowResident&&) = delete;

  [[nodiscard]] bool load(std::size_t action_dim, std::size_t cond_dim, std::size_t hidden,
                          std::size_t time_dim, std::size_t mlp_layers, const float* in_proj,
                          const float* time_proj, const float* cond_proj, const float* out_proj,
                          const std::array<const float*, kMaxMlp>& layers,
                          const float* freqs) noexcept;
  [[nodiscard]] bool begin(std::span<const float> cond, std::span<const float> x0) noexcept;
  [[nodiscard]] std::size_t advance(std::size_t step_budget, Method method, std::size_t steps,
                                    std::size_t next_step, std::span<float> out) noexcept;

private:
  void release() noexcept;
  [[nodiscard]] bool upload(const float* host, std::size_t count, float*& dst) noexcept;
  void velocity(float* x, float t, float* v) noexcept;

  std::size_t action_dim_{};
  std::size_t cond_dim_{};
  std::size_t hidden_{};
  std::size_t time_dim_{};
  std::size_t mlp_layers_{};
  float* in_proj_{};
  float* time_proj_{};
  float* cond_proj_{};
  float* out_proj_{};
  std::array<float*, kMaxMlp> layers_{};
  float* freqs_{};
  float* cond_{};
  float* c_emb_{};
  float* h_{};
  float* tmp_{};
  float* sinu_{};
  float* x_{};
  float* k1_{};
  float* k2_{};
  float* k3_{};
  float* k4_{};
  float* probe_{};
};

} // namespace fe
