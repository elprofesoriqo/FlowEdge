#include "engine.h"

#include "runtime/engine_runtime.h"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <optional>
#include <string_view>

namespace {

const char*& last_error()
{
  thread_local const char* error = "";
  return error;
}

[[nodiscard]] bool solver_method(int method, fe::FlowHead::Method& solver) noexcept
{
  if (method < FE_SOLVER_EULER || method > FE_SOLVER_RK4)
    return false;
  solver = (method == FE_SOLVER_RK4)    ? fe::FlowHead::kRK4
           : (method == FE_SOLVER_HEUN) ? fe::FlowHead::kHeun
                                        : fe::FlowHead::kEuler;
  return true;
}

[[nodiscard]] std::optional<unsigned> environment_thread_count(const char*& error) noexcept
{
  const char* const value = std::getenv("FLOWEDGE_THREADS");
  if (value == nullptr || value[0] == '\0')
    return fe::EngineRuntime::recommended_thread_count();
  const std::string_view text{value};
  unsigned threads{0u};
  const auto [end, code] = std::from_chars(text.data(), text.data() + text.size(), threads);
  if (code != std::errc{} || end != text.data() + text.size() ||
      threads > fe::EngineRuntime::kMaxPoolThreads) {
    error = "FLOWEDGE_THREADS must be an integer from 0 through 8";
    return std::nullopt;
  }
  return threads;
}

} // namespace

struct FeEngine
{
  FeEngine(const char* path, std::size_t slab_bytes, unsigned worker_threads, const char*& error)
      : runtime{path, slab_bytes, worker_threads, error}
  {
  }

  fe::EngineRuntime runtime;
};

namespace {

fe_engine* load_engine(const char* path, unsigned worker_threads)
{
  if (path == nullptr) {
    last_error() = "Invalid null model path";
    return nullptr;
  }
  if (worker_threads > fe::EngineRuntime::kMaxPoolThreads) {
    last_error() = "Worker thread count must be from 0 through 8";
    return nullptr;
  }
  try {
    const std::size_t slab_bytes = fe::EngineRuntime::required_slab_bytes(path);
    if (slab_bytes == 0uz) {
      last_error() = "Failed to load safetensors file or find required tensors";
      return nullptr;
    }
    auto engine = std::make_unique<FeEngine>(path, slab_bytes, worker_threads, last_error());
    if (!engine->runtime.valid()) {
      last_error() = "Model architecture initialization failed";
      return nullptr;
    }
    return engine.release();
  } catch (const std::exception&) {
    last_error() = "Exception during engine load (likely OOM)";
    return nullptr;
  }
}

} // namespace

const char* fe_engine_last_error(void)
{
  return last_error();
}

fe_engine* fe_engine_load(const char* path)
{
  last_error() = "";
  const std::optional<unsigned> worker_threads = environment_thread_count(last_error());
  return worker_threads ? load_engine(path, *worker_threads) : nullptr;
}

fe_engine* fe_engine_load_with_threads(const char* path, unsigned worker_threads)
{
  last_error() = "";
  return load_engine(path, worker_threads);
}

void fe_engine_free(fe_engine* engine)
{
  const std::unique_ptr<FeEngine> owner{engine}; // adopt + delete
}

void fe_engine_dims(const fe_engine* engine, std::size_t* d_model, std::size_t* n_layers)
{
  if (engine == nullptr)
    return;
  const fe::MambaConfig& config = engine->runtime.config();
  if (d_model != nullptr)
    *d_model = config.d_model;
  if (n_layers != nullptr)
    *n_layers = config.n_layers;
}

unsigned fe_engine_thread_count(const fe_engine* engine)
{
  return engine != nullptr ? engine->runtime.thread_count() : 0u;
}

#if defined(__MINGW32__) && defined(__AVX2__)
__attribute__((force_align_arg_pointer))
#endif
int fe_engine_run(fe_engine* engine, const std::int32_t* tokens, std::size_t seq_len, float* out)
{
  last_error() = "";
  if (engine == nullptr || tokens == nullptr || out == nullptr || seq_len == 0uz) {
    last_error() = "Invalid arguments to fe_engine_run";
    return 1;
  }
  return engine->runtime.run(tokens, seq_len, out, last_error());
}

#if defined(__MINGW32__) && defined(__AVX2__)
__attribute__((force_align_arg_pointer))
#endif
int fe_engine_step(fe_engine* engine, std::int32_t token, float* out)
{
  last_error() = "";
  if (engine == nullptr || out == nullptr) {
    last_error() = "Invalid null arguments to fe_engine_step";
    return 1;
  }
  return engine->runtime.step(token, out, last_error());
}

void fe_engine_reset(fe_engine* engine)
{
  if (engine != nullptr)
    engine->runtime.reset();
}

std::size_t fe_engine_action_dim(const fe_engine* engine)
{
  return engine != nullptr ? engine->runtime.action_dim() : 0uz;
}

std::size_t fe_engine_condition_dim(const fe_engine* engine)
{
  return engine != nullptr ? engine->runtime.condition_dim() : 0uz;
}

#if defined(__MINGW32__) && defined(__AVX2__)
__attribute__((force_align_arg_pointer))
#endif
int fe_engine_sample(fe_engine* engine, const std::int32_t* tokens, std::size_t seq_len,
                     const float* noise, std::size_t steps, int method, float* action)
{
  last_error() = "";
  if (engine == nullptr || tokens == nullptr || noise == nullptr || action == nullptr ||
      seq_len == 0uz || steps == 0uz) {
    last_error() = "Invalid arguments to fe_engine_sample";
    return 1;
  }
  fe::FlowHead::Method solver{};
  if (!solver_method(method, solver)) {
    last_error() = "Invalid solver method";
    return 5;
  }
  return engine->runtime.sample(tokens, seq_len, noise, steps, solver, action, last_error());
}

#if defined(__MINGW32__) && defined(__AVX2__)
__attribute__((force_align_arg_pointer))
#endif
int fe_engine_sample_condition(fe_engine* engine, const float* condition, const float* noise,
                               std::size_t steps, int method, float* action)
{
  last_error() = "";
  if (engine == nullptr || condition == nullptr || noise == nullptr || action == nullptr ||
      steps == 0uz) {
    last_error() = "Invalid arguments to fe_engine_sample_condition";
    return 1;
  }
  fe::FlowHead::Method solver{};
  if (!solver_method(method, solver)) {
    last_error() = "Invalid solver method";
    return 5;
  }
  return engine->runtime.sample_condition(condition, noise, steps, solver, action, last_error());
}

#if defined(__MINGW32__) && defined(__AVX2__)
__attribute__((force_align_arg_pointer))
#endif
int fe_engine_flow_begin(fe_engine* engine, const float* condition, const float* noise,
                         std::size_t steps, int method)
{
  last_error() = "";
  if (engine == nullptr || condition == nullptr || noise == nullptr || steps == 0uz) {
    last_error() = "Invalid arguments to fe_engine_flow_begin";
    return 1;
  }
  fe::FlowHead::Method solver{};
  if (!solver_method(method, solver)) {
    last_error() = "Invalid solver method";
    return 5;
  }
  return engine->runtime.flow_begin(condition, noise, steps, solver, last_error());
}

#if defined(__MINGW32__) && defined(__AVX2__)
__attribute__((force_align_arg_pointer))
#endif
int fe_engine_flow_advance(fe_engine* engine, std::size_t step_budget, float* action,
                           std::size_t* steps_remaining)
{
  last_error() = "";
  if (engine == nullptr || action == nullptr || step_budget == 0uz) {
    last_error() = "Invalid arguments to fe_engine_flow_advance";
    return 1;
  }
  std::size_t remaining{0uz};
  const int rc = engine->runtime.flow_advance(step_budget, action, remaining, last_error());
  if (rc == 0 && steps_remaining != nullptr)
    *steps_remaining = remaining;
  return rc;
}

std::size_t fe_engine_decode_state_bytes(const fe_engine* engine)
{
  return engine != nullptr ? engine->runtime.decode_state_bytes() : 0uz;
}

int fe_engine_export_decode_state(const fe_engine* engine, void* destination, std::size_t bytes)
{
  last_error() = "";
  if (engine == nullptr || destination == nullptr) {
    last_error() = "Invalid arguments to fe_engine_export_decode_state";
    return 1;
  }
  return engine->runtime.export_decode_state({static_cast<std::byte*>(destination), bytes},
                                             last_error());
}

int fe_engine_import_decode_state(fe_engine* engine, const void* source, std::size_t bytes)
{
  last_error() = "";
  if (engine == nullptr || source == nullptr) {
    last_error() = "Invalid arguments to fe_engine_import_decode_state";
    return 1;
  }
  return engine->runtime.import_decode_state({static_cast<const std::byte*>(source), bytes},
                                             last_error());
}
