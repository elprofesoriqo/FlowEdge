#include "engine.h"

#include "runtime/engine_runtime.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>

namespace {

const char*& last_error()
{
  thread_local const char* error = "";
  return error;
}

} // namespace

struct FeEngine
{
  FeEngine(const char* path, std::size_t slab_bytes, const char*& error)
      : runtime{path, slab_bytes, error}
  {
  }

  fe::EngineRuntime runtime;
};

const char* fe_engine_last_error(void)
{
  return last_error();
}

fe_engine* fe_engine_load(const char* path)
{
  last_error() = "";
  if (path == nullptr) {
    last_error() = "Invalid null model path";
    return nullptr;
  }
  try {
    const std::size_t slab_bytes = fe::EngineRuntime::required_slab_bytes(path);
    if (slab_bytes == 0uz) {
      last_error() = "Failed to load safetensors file or find required tensors";
      return nullptr;
    }
    auto engine = std::make_unique<FeEngine>(path, slab_bytes, last_error());
    if (!engine->runtime.valid()) {
      last_error() = "Model architecture initialization failed";
      return nullptr;
    }
    if (!engine->runtime.has_compatible_flow_head()) {
      last_error() = "Flow conditioning dimension does not match the backbone";
      return nullptr;
    }
    return engine.release();
  } catch (const std::exception&) {
    last_error() = "Exception during engine load (likely OOM)";
    return nullptr;
  }
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
  if (engine == nullptr || tokens == nullptr || out == nullptr) {
    last_error() = "Invalid null arguments to fe_engine_run";
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
  if (method < FE_SOLVER_EULER || method > FE_SOLVER_RK4) {
    last_error() = "Invalid solver method";
    return 5;
  }
  const auto solver = (method == FE_SOLVER_RK4)    ? fe::FlowHead::kRK4
                      : (method == FE_SOLVER_HEUN) ? fe::FlowHead::kHeun
                                                   : fe::FlowHead::kEuler;
  return engine->runtime.sample(tokens, seq_len, noise, steps, solver, action, last_error());
}
