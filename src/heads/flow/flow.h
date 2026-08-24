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

  FlowHead(std::span<const TensorView> weights, Arena& scratch) noexcept;

  [[nodiscard]] bool valid() const noexcept { return ok_; }
  [[nodiscard]] const FlowConfig& config() const noexcept { return cfg_; }

  void set_pool(ThreadPool* p) noexcept { pool_ = p; }

  // x0: noise [action_dim]
  // cond: [cond_dim]
  // out: action [action_dim]
  void sample(std::span<const float> cond, std::span<const float> x0, std::size_t steps,
              Method method, std::span<float> out) noexcept;

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
  const uint16_t* in_proj_{};                     // [hidden][action_dim]
  const uint16_t* time_proj_{};                   // [hidden][time_dim]
  const uint16_t* cond_proj_{};                   // [hidden][cond_dim]
  std::array<const uint16_t*, kMaxMlp> layers_{}; // [hidden][hidden]
  const uint16_t* out_proj_{};                    // [action_dim][hidden]
  const float* freqs_{};                          // [time_dim/2] computed at init
  Arena* scratch_{};
  ThreadPool* pool_{nullptr};
  bool ok_{false};
};

} // namespace fe
