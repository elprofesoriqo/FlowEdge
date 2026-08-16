#include "engine.h"

#include "arena/arena.h"
#include "heads/flow/flow.h"
#include "loader/safetensors.h"
#include "models/mamba/mamba.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace {

const char*& get_last_error()
{
  thread_local const char* err = "";
  return err;
}

constexpr std::size_t max_tensors = 1024uz;
constexpr std::size_t k_max_decode_seq = 512uz; // scratch is sized for prefills up to this length

std::size_t slab_bytes(const char* path)
{
  // slab = W_bytes_stored + k_max_decode_seq * (3*d_state*d_inner + 16*d_inner + 8*d_model) * 4
  // W_bytes_stored is 2 B/elem for BF16, 4 B/elem for F32
  // scratch term is always float
  const std::size_t weights = fe::safetensors_weight_bytes(path); // BF16 = 2B/elem
  if (weights == 0uz)
    return 0uz;
  const auto emb = fe::safetensors_tensor_shape(path, "backbone.embeddings.weight");
  const auto a_log = fe::safetensors_tensor_shape(path, "backbone.layers.0.mixer.A_log");
  const std::size_t d_model = emb[1];
  const std::size_t d_inner = a_log[0];
  const std::size_t d_state = a_log[1];
  if (d_model == 0uz || d_inner == 0uz || d_state == 0uz) // malformed header
    return 0uz;
  const std::size_t per_token = (3uz * d_state * d_inner) + (16uz * d_inner) + (8uz * d_model);
  return weights + (k_max_decode_seq * per_token * sizeof(float));
}

std::size_t load_views(const char* path, fe::Arena& arena, std::span<fe::TensorView> views) noexcept
{
  std::size_t count{0uz};
  return fe::load_safetensors(path, arena, views, count) ? count : 0uz;
}

} // namespace

struct FeEngine
{
  std::size_t slab_size; // stored so the span can be built before arena init
  std::vector<std::byte> slab;
  fe::Arena arena;
  std::array<fe::TensorView, max_tensors> views{};
  std::size_t n;
  fe::Mamba model;
  fe::FlowHead flow;
  std::span<float> dstate; // persistent streaming state

  FeEngine(const char* path, std::size_t slab_sz)
      : slab_size{slab_sz}, slab(slab_size), arena{std::span<std::byte>{slab.data(), slab_size}},
        n{load_views(path, arena, views)},
        model{std::span<const fe::TensorView>{views.data(), n}, arena},
        flow{std::span<const fe::TensorView>{views.data(), n}, arena}
  {
    if (model.valid())
      if (auto* const p = arena.alloc_array<float>(model.state_size(), fe::kSimdAlign))
        dstate = {p, model.state_size()};
  }

  // Embed tokens and run the backbone into `hidden` [seq_len*d_model]
  // Returns 0, or 2 (arena exhausted) / 3 (token out of range).
  int run_backbone(const std::int32_t* tokens, std::size_t seq_len, float* hidden) noexcept
  {
    const fe::MambaConfig& c = model.config();
    const std::size_t hz = seq_len * c.d_model;
    auto* const in = arena.alloc_array<float>(hz, fe::kSimdAlign);
    if (in == nullptr) {
      get_last_error() = "Arena exhausted during backbone forward pass";
      return 2;
    }
    const float* const emb = model.embedding();
    for (std::size_t t{0uz}; t < seq_len; ++t) {
      const std::int32_t tok = tokens[t];
      if (std::cmp_less(tok, 0) || std::cmp_greater_equal(tok, c.vocab)) {
        get_last_error() = "Token ID out of vocabulary range";
        return 3;
      }
      const auto utok = static_cast<std::size_t>(tok);
      std::copy_n(emb + (utok * c.d_model), c.d_model, in + (t * c.d_model));
    }
    model.forward({in, hz}, {hidden, hz}, seq_len);
    return 0;
  }
};

const char* fe_engine_last_error(void)
{
  return get_last_error();
}

fe_engine* fe_engine_load(const char* path)
{
  get_last_error() = "";
  try {
    const std::size_t slab_sz = slab_bytes(path); // one header pass; reused as the arena size
    if (slab_sz == 0uz) {
      get_last_error() = "Failed to load safetensors file or find required tensors";
      return nullptr;
    }
    auto engine = std::make_unique<FeEngine>(path, slab_sz);
    if (!engine->model.valid()) {
      get_last_error() = "Model architecture initialization failed";
      return nullptr;
    }
    return engine.release();
  } catch (const std::exception& e) {
    get_last_error() = "Exception during engine load (likely OOM)";
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
  const fe::MambaConfig& c = engine->model.config();
  if (d_model != nullptr)
    *d_model = c.d_model;
  if (n_layers != nullptr)
    *n_layers = c.n_layers;
}

#if defined(__MINGW32__) && defined(__AVX2__)
__attribute__((force_align_arg_pointer))
#endif
int fe_engine_run(fe_engine* engine, const std::int32_t* tokens, std::size_t seq_len, float* out)
{
  get_last_error() = "";
  if (engine == nullptr || tokens == nullptr || out == nullptr) {
    get_last_error() = "Invalid null arguments to fe_engine_run";
    return 1;
  }
  std::byte* const mark = engine->arena.mark();
  const int rc = engine->run_backbone(tokens, seq_len, out);
  engine->arena.reset_to(mark);
  return rc;
}

#if defined(__MINGW32__) && defined(__AVX2__)
__attribute__((force_align_arg_pointer))
#endif
int fe_engine_step(fe_engine* engine, std::int32_t token, float* out)
{
  get_last_error() = "";
  if (engine == nullptr || out == nullptr) {
    get_last_error() = "Invalid null arguments to fe_engine_step";
    return 1;
  }
  if (engine->dstate.empty()) [[unlikely]] { // model invalid / no state
    get_last_error() = "Model has no streaming state initialized";
    return 2;
  }
  const fe::MambaConfig& c = engine->model.config();
  if (std::cmp_less(token, 0) || std::cmp_greater_equal(token, c.vocab)) {
    get_last_error() = "Token ID out of vocabulary range";
    return 3;
  }
  fe::Arena& arena = engine->arena;
  std::byte* const mark = arena.mark();
  auto* const x = arena.alloc_array<float>(c.d_model, fe::kSimdAlign);
  if (x == nullptr) {
    arena.reset_to(mark);
    get_last_error() = "Arena exhausted during streaming step";
    return 2;
  }
  const float* const emb = engine->model.embedding();
  const auto utok = static_cast<std::size_t>(token);
  std::copy_n(emb + (utok * c.d_model), c.d_model, x);
  engine->model.decode({x, c.d_model}, engine->dstate, {out, c.d_model});
  arena.reset_to(mark);
  return 0;
}

void fe_engine_reset(fe_engine* engine)
{
  if (engine != nullptr)
    std::ranges::fill(engine->dstate, 0.0F); // begin a fresh sequence
}

std::size_t fe_engine_action_dim(const fe_engine* engine)
{
  return (engine != nullptr && engine->flow.valid()) ? engine->flow.config().action_dim : 0uz;
}

#if defined(__MINGW32__) && defined(__AVX2__)
__attribute__((force_align_arg_pointer))
#endif
int fe_engine_sample(fe_engine* engine, const std::int32_t* tokens, std::size_t seq_len,
                     const float* noise, std::size_t steps, int method, float* action)
{
  get_last_error() = "";
  if (engine == nullptr || tokens == nullptr || noise == nullptr || action == nullptr ||
      seq_len == 0uz || steps == 0uz) {
    get_last_error() = "Invalid arguments to fe_engine_sample";
    return 1;
  }
  if (!engine->flow.valid()) {
    get_last_error() = "Model has no flow head to sample from";
    return 4;
  }
  fe::Arena& arena = engine->arena;
  std::byte* const mark = arena.mark();
  const fe::MambaConfig& c = engine->model.config();
  auto* const hidden = arena.alloc_array<float>(seq_len * c.d_model, fe::kSimdAlign);
  if (hidden == nullptr) {
    arena.reset_to(mark);
    get_last_error() = "Arena exhausted allocating hidden states";
    return 2;
  }
  const int rc = engine->run_backbone(tokens, seq_len, hidden);
  if (rc != 0) {
    arena.reset_to(mark);
    return rc; // g_last_error set by run_backbone
  }
  const std::size_t a = engine->flow.config().action_dim;
  const std::span<const float> cond{hidden + ((seq_len - 1uz) * c.d_model), c.d_model};
  const auto m = (method == 2)   ? fe::FlowHead::kRK4
                 : (method == 1) ? fe::FlowHead::kHeun
                                 : fe::FlowHead::kEuler;
  engine->flow.sample(cond, {noise, a}, steps, m, {action, a});
  arena.reset_to(mark);
  return 0;
}