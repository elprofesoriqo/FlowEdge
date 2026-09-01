#pragma once

#include "arena/arena.h"
#include "arena/thread_pool.h"
#include "heads/flow/flow.h"
#include "loader/safetensors.h"
#include "models/mamba/mamba.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <thread>
#include <vector>

namespace fe {

// Owns the allocation-free policy runtime behind the C ABI.
class EngineRuntime
{
public:
  [[nodiscard]] static std::size_t required_slab_bytes(std::string_view path) noexcept;

  EngineRuntime(std::string_view path, std::size_t slab_bytes, const char*& error);
  ~EngineRuntime();

  EngineRuntime(const EngineRuntime&) = delete;
  EngineRuntime(EngineRuntime&&) = delete;
  EngineRuntime& operator=(const EngineRuntime&) = delete;
  EngineRuntime& operator=(EngineRuntime&&) = delete;

  [[nodiscard]] bool valid() const noexcept { return model_.valid() || flow_.valid(); }
  [[nodiscard]] bool has_compatible_flow_head() const noexcept;
  [[nodiscard]] const MambaConfig& config() const noexcept { return model_.config(); }
  [[nodiscard]] unsigned thread_count() const noexcept;
  [[nodiscard]] std::size_t action_dim() const noexcept;
  [[nodiscard]] std::size_t condition_dim() const noexcept;
  [[nodiscard]] std::size_t decode_state_bytes() const noexcept
  {
    return decode_state_.size_bytes();
  }

  int run(const std::int32_t* tokens, std::size_t seq_len, float* out, const char*& error) noexcept;
  int step(std::int32_t token, float* out, const char*& error) noexcept;
  void reset() noexcept;
  int sample(const std::int32_t* tokens, std::size_t seq_len, const float* noise, std::size_t steps,
             FlowHead::Method method, float* action, const char*& error) noexcept;
  int sample_condition(const float* condition, const float* noise, std::size_t steps,
                       FlowHead::Method method, float* action, const char*& error) noexcept;
  int flow_begin(const float* condition, const float* noise, std::size_t steps,
                 FlowHead::Method method, const char*& error) noexcept;
  int flow_advance(std::size_t step_budget, float* action, std::size_t& steps_remaining,
                   const char*& error) noexcept;
  int export_decode_state(std::span<std::byte> destination, const char*& error) const noexcept;
  int import_decode_state(std::span<const std::byte> source, const char*& error) noexcept;

private:
  static constexpr std::size_t kMaxTensors = 1024uz;
  static constexpr unsigned kMaxPoolThreads = 8u;
  static constexpr std::size_t kThreadRingSlots = 128uz;

  [[nodiscard]] std::size_t load_views(std::string_view path, const char*& error) noexcept;
  int run_backbone(const std::int32_t* tokens, std::size_t seq_len, float* hidden,
                   const char*& error) noexcept;

  std::vector<std::byte> slab_;
  Arena arena_;
  std::array<TensorView, kMaxTensors> views_{};
  std::size_t tensor_count_{};
  Mamba model_;
  FlowHead flow_;
  ThreadPool* pool_{};
  std::span<std::jthread> workers_{};
  std::span<float> decode_state_{};
  std::span<float> flow_workspace_{};
  FlowHead::SamplerState flow_state_{};
};

} // namespace fe
