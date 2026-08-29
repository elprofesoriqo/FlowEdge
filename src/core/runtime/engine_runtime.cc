#include "runtime/engine_runtime.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <ranges>
#include <thread>
#include <utility>

namespace fe {
namespace {

constexpr std::size_t k_max_decode_seq = 512uz; // scratch is sized for prefills up to this length

} // namespace

std::size_t EngineRuntime::required_slab_bytes(std::string_view path) noexcept
{
  const std::size_t weights = safetensors_weight_bytes(path);
  if (weights == 0uz)
    return 0uz;

  const auto emb = safetensors_tensor_shape(path, "backbone.embeddings.weight");
  const auto a_log = safetensors_tensor_shape(path, "backbone.layers.0.mixer.A_log");
  const auto conv = safetensors_tensor_shape(path, "backbone.layers.0.mixer.conv1d.weight");
  const auto flow_in = safetensors_tensor_shape(path, "flow.in_proj.weight");
  const auto time_proj = safetensors_tensor_shape(path, "flow.time_proj.weight");
  const std::size_t d_model = emb[1];
  const std::size_t d_inner = a_log[0];
  const std::size_t d_state = a_log[1];
  const std::size_t d_conv = conv[2];
  const std::size_t flow_hidden = flow_in[0];
  const std::size_t action_dim = flow_in[1];
  const std::size_t flow_time_dim = time_proj[1];
  const bool has_backbone = d_model != 0uz && d_inner != 0uz && d_state != 0uz && d_conv != 0uz;
  const bool has_flow = flow_hidden != 0uz && action_dim != 0uz && flow_time_dim != 0uz;
  if (!has_backbone && !has_flow)
    return 0uz;

  const std::size_t per_token =
      has_backbone ? (3uz * d_state * d_inner) + (16uz * d_inner) + (8uz * d_model) : 0uz;
  const std::size_t persistent_state =
      has_backbone ? 64uz * d_inner * (d_conv + (2uz * d_state)) * sizeof(float) : 0uz;
  const std::size_t flow_floats =
      has_flow ? (3uz * flow_hidden) + (6uz * action_dim) + ((3uz * flow_time_dim) / 2uz) : 0uz;
  const std::size_t runtime = (kThreadRingSlots * sizeof(Task)) +
                              (kThreadRingSlots * sizeof(std::size_t)) +
                              (kMaxPoolThreads * sizeof(std::jthread)) + sizeof(ThreadPool) +
                              (flow_floats * sizeof(float)) + persistent_state + 4096uz;
  return weights + (k_max_decode_seq * per_token * sizeof(float)) + runtime;
}

unsigned EngineRuntime::recommended_thread_count() noexcept
{
  const unsigned hardware_threads = std::thread::hardware_concurrency();
  if (hardware_threads < 2u)
    return 0u;
  // Short-sequence policy inference is normally memory-bandwidth-bound. Four
  // workers saturate typical desktop/edge memory without the WSL/SMT cliff
  // observed when every logical CPU is used.
  return std::min(std::max(hardware_threads / 2u, 2u), 4u);
}

EngineRuntime::EngineRuntime(std::string_view path, std::size_t slab_bytes, unsigned worker_threads,
                             const char*& error)
    : slab_{slab_bytes}, arena_{std::span<std::byte>{slab_.data(), slab_.size()}},
      tensor_count_{load_views(path, error)},
      model_{std::span<const TensorView>{views_.data(), tensor_count_}, arena_},
      flow_{std::span<const TensorView>{views_.data(), tensor_count_}, arena_}
{
  if (worker_threads > 0u) {
    auto* ring = arena_.alloc_array<Task, kSimdAlign>(kThreadRingSlots);
    auto* sequence = arena_.alloc_array<std::size_t, kSimdAlign>(kThreadRingSlots);
    auto* worker_mem = static_cast<std::jthread*>(
        arena_.alloc<alignof(std::jthread)>(sizeof(std::jthread) * worker_threads));
    auto* pool_mem = arena_.alloc<alignof(ThreadPool)>(sizeof(ThreadPool));
    if (ring != nullptr && sequence != nullptr && worker_mem != nullptr && pool_mem != nullptr) {
      workers_ = {worker_mem, worker_threads};
      for (unsigned i{0u}; i < worker_threads; ++i)
        std::construct_at(&workers_[i]);
      pool_ = std::construct_at(static_cast<ThreadPool*>(pool_mem),
                                std::span<Task>{ring, kThreadRingSlots},
                                std::span<std::size_t>{sequence, kThreadRingSlots}, workers_,
                                worker_threads);
    }
  }

  model_.set_pool(pool_);
  flow_.set_pool(pool_);
  if (model_.valid())
    if (auto* const state = arena_.alloc_array<float, kSimdAlign>(model_.state_size()))
      decode_state_ = {state, model_.state_size()};
  if (flow_.valid())
    if (auto* const workspace =
            arena_.alloc_array<float, kSimdAlign>(flow_.sampler_workspace_size()))
      flow_workspace_ = {workspace, flow_.sampler_workspace_size()};
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
  return !flow_.valid() || !model_.valid() || flow_.config().cond_dim == model_.config().d_model;
}

unsigned EngineRuntime::thread_count() const noexcept
{
  return pool_ != nullptr ? pool_->nthreads() : 0u;
}

std::size_t EngineRuntime::action_dim() const noexcept
{
  return flow_.valid() ? flow_.config().action_dim : 0uz;
}

std::size_t EngineRuntime::condition_dim() const noexcept
{
  return flow_.valid() ? flow_.config().cond_dim : 0uz;
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
  if (!model_.valid()) {
    error = "Model has no backbone for token-conditioned sampling";
    return 4;
  }
  if (!has_compatible_flow_head()) {
    error = "Flow conditioning dimension does not match the backbone";
    return 4;
  }
  if (seq_len == 0uz || seq_len > k_max_decode_seq) {
    error = "Token sequence exceeds the configured prefill limit";
    return 1;
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

  const std::span<const float> cond{hidden + ((seq_len - 1uz) * c.d_model), c.d_model};
  int sample_rc = flow_begin(cond.data(), noise, steps, method, error);
  if (sample_rc == 0) {
    std::size_t remaining{steps};
    sample_rc = flow_advance(steps, action, remaining, error);
  }
  arena_.reset_to(mark);
  return sample_rc;
}

int EngineRuntime::sample_condition(const float* condition, const float* noise, std::size_t steps,
                                    FlowHead::Method method, float* action,
                                    const char*& error) noexcept
{
  if (!flow_.valid()) {
    error = "Model has no flow head to sample from";
    return 4;
  }
  const int rc = flow_begin(condition, noise, steps, method, error);
  if (rc != 0)
    return rc;
  std::size_t remaining{steps};
  return flow_advance(steps, action, remaining, error);
}

int EngineRuntime::flow_begin(const float* condition, const float* noise, std::size_t steps,
                              FlowHead::Method method, const char*& error) noexcept
{
  if (!flow_.valid()) {
    error = "Model has no flow head to sample from";
    return 4;
  }
  if (flow_workspace_.empty()) [[unlikely]] {
    error = "Flow sampler workspace is unavailable";
    return 2;
  }
  if (!flow_.sampler_begin({condition, flow_.config().cond_dim}, {noise, flow_.config().action_dim},
                           steps, method, flow_workspace_, flow_state_)) [[unlikely]] {
    error = "Failed to initialize resumable flow sampler";
    return 2;
  }
  return 0;
}

int EngineRuntime::flow_advance(std::size_t step_budget, float* action,
                                std::size_t& steps_remaining, const char*& error) noexcept
{
  if (!flow_state_.active) {
    error = "No resumable flow sample is active";
    return 6;
  }
  static_cast<void>(
      flow_.sampler_advance(flow_state_, step_budget, {action, flow_.config().action_dim}));
  steps_remaining = flow_state_.remaining();
  return 0;
}

int EngineRuntime::export_decode_state(std::span<std::byte> destination,
                                       const char*& error) const noexcept
{
  const std::span<const std::byte> state = std::as_bytes(decode_state_);
  if (state.empty()) {
    error = "Model has no streaming decode state";
    return 7;
  }
  if (destination.size() < state.size()) {
    error = "Decode state destination is too small";
    return 1;
  }
  std::ranges::copy(state, destination.begin());
  return 0;
}

int EngineRuntime::import_decode_state(std::span<const std::byte> source,
                                       const char*& error) noexcept
{
  const std::span<std::byte> state = std::as_writable_bytes(decode_state_);
  if (state.empty()) {
    error = "Model has no streaming decode state";
    return 7;
  }
  if (source.size() != state.size()) {
    error = "Decode state snapshot has the wrong size";
    return 1;
  }
  std::ranges::copy(source, state.begin());
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
  if (!model_.valid()) {
    error = "Model has no backbone";
    return 4;
  }
  const MambaConfig& c = model_.config();
  if (seq_len == 0uz || seq_len > k_max_decode_seq ||
      seq_len > (std::numeric_limits<std::size_t>::max() / c.d_model)) {
    error = "Token sequence length is outside the configured prefill limit";
    return 1;
  }
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
