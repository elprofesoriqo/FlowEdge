#pragma once

#include "arena/arena.h"
#include "loader/safetensors.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace fe {

class ThreadPool;

struct FlowConfig
{
  std::size_t action_dim{}, cond_dim{}, hidden{}, time_dim{}, mlp_layers{};
};

// action head: dx/dt = v(x,t,cond) from noise to an action
// cond is the encoder Mamba hidden state
class FlowHead
{
public:
  static constexpr std::size_t kMaxMlp = 8uz;
  enum Method : int
  {
    kEuler = 0,
    kHeun = 1,
    kRK4 = 2
  };

  struct SamplerState
  {
    std::span<float> condition_embedding{};
    std::span<float> x{};
    std::span<float> k1{};
    std::span<float> k2{};
    std::span<float> k3{};
    std::span<float> k4{};
    std::span<float> probe{};
    std::size_t steps{};
    std::size_t next_step{};
    Method method{kEuler};
    bool active{false};

    [[nodiscard]] std::size_t remaining() const noexcept
    {
      return active && next_step < steps ? steps - next_step : 0uz;
    }
  };

  FlowHead(std::span<const TensorView> weights, Arena& scratch) noexcept;

  [[nodiscard]] bool valid() const noexcept { return ok_; }
  [[nodiscard]] const FlowConfig& config() const noexcept { return cfg_; }
  [[nodiscard]] std::size_t sampler_workspace_size() const noexcept
  {
    return cfg_.hidden + (6uz * cfg_.action_dim);
  }

  void set_pool(ThreadPool* p) noexcept { pool_ = p; }

  // x0: noise [action_dim]
  // cond: [cond_dim]
  // out: action [action_dim]
  void sample(std::span<const float> cond, std::span<const float> x0, std::size_t steps,
              Method method, std::span<float> out) noexcept;

  // Start and cooperatively advance a solver without allocating. The caller
  // owns `workspace` for the entire session and may execute a bounded number
  // of complete solver steps per control-loop tick.
  [[nodiscard]] bool sampler_begin(std::span<const float> cond, std::span<const float> x0,
                                   std::size_t steps, Method method, std::span<float> workspace,
                                   SamplerState& state) noexcept;
  [[nodiscard]] std::size_t sampler_advance(SamplerState& state, std::size_t step_budget,
                                            std::span<float> out) noexcept;

private:
  // v[action_dim] = velocity(x[action_dim], t, c_emb[hidden])
  // c_emb = cond_proj·cond
  void velocity(std::span<const float> x, float t, std::span<const float> c_emb,
                std::span<float> v) noexcept;
  [[nodiscard]] std::span<float> arena_span(std::size_t n) noexcept
  {
    return scratch_->alloc_span<float, kSimdAlign>(n);
  }

  FlowConfig cfg_{};
  WeightView in_proj_{};                     // [hidden][action_dim]
  WeightView time_proj_{};                   // [hidden][time_dim]
  WeightView cond_proj_{};                   // [hidden][cond_dim]
  std::array<WeightView, kMaxMlp> layers_{}; // [hidden][hidden]
  WeightView out_proj_{};                    // [action_dim][hidden]
  const float* freqs_{};                     // [time_dim/2] computed at init
  Arena* scratch_{};
  ThreadPool* pool_{nullptr};
  bool ok_{false};
};

} // namespace fe
