#include "runtime/engine_runtime.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ranges>
#include <thread>
#include <utility>

namespace fe {
namespace {

constexpr std::size_t kMaxDecodeSeq = 512uz; // scratch is sized for prefills up to this length

} // namespace

std::size_t EngineRuntime::required_slab_bytes(std::string_view path) noexcept
{
  const std::size_t weights = safetensors_weight_bytes(path);
  if (weights == 0uz)
    return 0uz;

  const auto emb = safetensors_tensor_shape(path, "backbone.embeddings.weight");
  const auto a_log = safetensors_tensor_shape(path, "backbone.layers.0.mixer.A_log");
  const auto conv = safetensors_tensor_shape(path, "backbone.layers.0.mixer.conv1d.weight");
  const auto time_proj = safetensors_tensor_shape(path, "flow.time_proj.weight");
  const std::size_t d_model = emb[1];
  const std::size_t d_inner = a_log[0];
  const std::size_t d_state = a_log[1];
  const std::size_t d_conv = conv[2];
  const std::size_t flow_time_dim = time_proj[1];
  if (d_model == 0uz || d_inner == 0uz || d_state == 0uz || d_conv == 0uz)
    return 0uz;

  const std::size_t per_token = (3uz * d_state * d_inner) + (16uz * d_inner) + (8uz * d_model);
  const std::size_t persistent_state = 64uz * d_inner * (d_conv + d_state) * sizeof(float);
  const std::size_t runtime = (kThreadRingSlots * sizeof(Task)) +
                              (kThreadRingSlots * sizeof(std::size_t)) +
                              (kMaxPoolThreads * sizeof(std::jthread)) + sizeof(ThreadPool) +
                              (flow_time_dim / 2uz * sizeof(float)) + persistent_state + 4096uz;
  return weights + (kMaxDecodeSeq * per_token * sizeof(float)) + runtime;
}

EngineRuntime::EngineRuntime(std::string_view path, std::size_t slab_bytes, const char*& error)
    : slab_{slab_bytes}, arena_{std::span<std::byte>{slab_.data(), slab_.size()}},
      tensor_count_{load_views(path, error)},
      model_{std::span<const TensorView>{views_.data(), tensor_count_}, arena_},
      flow_{std::span<const TensorView>{views_.data(), tensor_count_}, arena_}
{
  const unsigned hw_threads = std::thread::hardware_concurrency();
  const unsigned nthreads = std::min(hw_threads > 0 ? hw_threads : 1u, kMaxPoolThreads);

  auto* ring = arena_.alloc_array<Task, kSimdAlign>(kThreadRingSlots);
  auto* sequence = arena_.alloc_array<std::size_t, kSimdAlign>(kThreadRingSlots);
  auto* worker_mem = static_cast<std::jthread*>(
      arena_.alloc<alignof(std::jthread)>(sizeof(std::jthread) * nthreads));
  auto* pool_mem = arena_.alloc<alignof(ThreadPool)>(sizeof(ThreadPool));
  if (ring != nullptr && sequence != nullptr && worker_mem != nullptr && pool_mem != nullptr) {
    workers_ = {worker_mem, nthreads};
    for (unsigned i{0u}; i < nthreads; ++i)
      std::construct_at(&workers_[i]);
    pool_ =
        std::construct_at(static_cast<ThreadPool*>(pool_mem),
                          std::span<Task>{ring, kThreadRingSlots},
                          std::span<std::size_t>{sequence, kThreadRingSlots}, workers_, nthreads);
  }

  model_.set_pool(pool_);
  flow_.set_pool(pool_);
  if (model_.valid())
    if (auto* const state = arena_.alloc_array<float, kSimdAlign>(model_.state_size()))
      decode_state_ = {state, model_.state_size()};
}

EngineRuntime::~EngineRuntime()
{
  if (pool_ != nullptr)
    pool_->~ThreadPool();
  for (std::jthread& worker : workers_)
    std::destroy_at(&worker);
}

bool EngineRuntime::has_compatible_flow_head() const noexcept
{
  return !flow_.valid() || flow_.config().cond_dim == model_.config().d_model;
}

unsigned EngineRuntime::thread_count() const noexcept
{
  return pool_ != nullptr ? pool_->nthreads() : 0u;
}

std::size_t EngineRuntime::action_dim() const noexcept
{
  return flow_.valid() ? flow_.config().action_dim : 0uz;
}

int EngineRuntime::run(const std::int32_t* tokens, std::size_t seq_len, float* out,
                       const char*& error) noexcept
{
  std::byte* const mark = arena_.mark();
  const int rc = run_backbone(tokens, seq_len, out, error);
  arena_.reset_to(mark);
  return rc;
}

int EngineRuntime::step(std::int32_t token, float* out, const char*& error) noexcept
{
  if (decode_state_.empty()) [[unlikely]] {
    error = "Model has no streaming state initialized";
    return 2;
  }
  const MambaConfig& c = model_.config();
  if (std::cmp_less(token, 0) || std::cmp_greater_equal(token, c.vocab)) {
    error = "Token ID out of vocabulary range";
    return 3;
  }

  std::byte* const mark = arena_.mark();
  auto* const x = arena_.alloc_array<float, kSimdAlign>(c.d_model);
  if (x == nullptr) {
    error = "Arena exhausted during streaming step";
    return 2;
  }
  const auto token_index = static_cast<std::size_t>(token);
  std::copy_n(model_.embedding() + (token_index * c.d_model), c.d_model, x);
  model_.decode({x, c.d_model}, decode_state_, {out, c.d_model});
  arena_.reset_to(mark);
  return 0;
}

void EngineRuntime::reset() noexcept
{
  std::ranges::fill(decode_state_, 0.0F);
}

int EngineRuntime::sample(const std::int32_t* tokens, std::size_t seq_len, const float* noise,
                          std::size_t steps, FlowHead::Method method, float* action,
                          const char*& error) noexcept
{
  if (!flow_.valid()) {
    error = "Model has no flow head to sample from";
    return 4;
  }

  std::byte* const mark = arena_.mark();
  const MambaConfig& c = model_.config();
  auto* const hidden = arena_.alloc_array<float, kSimdAlign>(seq_len * c.d_model);
  if (hidden == nullptr) {
    error = "Arena exhausted allocating hidden states";
    return 2;
  }
  const int rc = run_backbone(tokens, seq_len, hidden, error);
  if (rc != 0) {
    arena_.reset_to(mark);
    return rc;
  }

  const std::size_t action_size = flow_.config().action_dim;
  const std::span<const float> cond{hidden + ((seq_len - 1uz) * c.d_model), c.d_model};
  flow_.sample(cond, {noise, action_size}, steps, method, {action, action_size});
  arena_.reset_to(mark);
  return 0;
}

std::size_t EngineRuntime::load_views(std::string_view path, const char*& error) noexcept
{
  std::size_t count{0uz};
  auto result = load_safetensors(path, arena_, views_, count);
  if (!result) {
    error = result.error();
    return 0uz;
  }
  return count;
}

int EngineRuntime::run_backbone(const std::int32_t* tokens, std::size_t seq_len, float* hidden,
                                const char*& error) noexcept
{
  const MambaConfig& c = model_.config();
  const std::size_t hidden_size = seq_len * c.d_model;
  auto* const input = arena_.alloc_array<float, kSimdAlign>(hidden_size);
  if (input == nullptr) {
    error = "Arena exhausted during backbone forward pass";
    return 2;
  }
  for (std::size_t t{0uz}; t < seq_len; ++t) {
    const std::int32_t token = tokens[t];
    if (std::cmp_less(token, 0) || std::cmp_greater_equal(token, c.vocab)) {
      error = "Token ID out of vocabulary range";
      return 3;
    }
    const auto token_index = static_cast<std::size_t>(token);
    std::copy_n(model_.embedding() + (token_index * c.d_model), c.d_model, input + (t * c.d_model));
  }
  model_.forward({input, hidden_size}, {hidden, hidden_size}, seq_len);
  return 0;
}

} // namespace fe
