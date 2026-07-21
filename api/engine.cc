#include "engine.h"

#include "../arena/arena.h"
#include "../loader/safetensors.h"
#include "../model/mamba.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t max_tensors = 1024uz;
constexpr std::size_t scratch_margin = std::size_t{128} << 20; // forward buffers above the weights

std::size_t slab_bytes(const char* path)
{
  std::error_code ec;
  const auto sz = std::filesystem::file_size(path, ec);
  return ec ? 0uz : static_cast<std::size_t>(sz) + scratch_margin;
}

} // namespace

struct FeEngine
{
  std::vector<std::byte> slab;
  fe::Arena arena;
  std::array<fe::TensorView, max_tensors> views{};
  std::size_t n;
  fe::Mamba model;

  explicit FeEngine(const char* path)
      : slab(slab_bytes(path)), arena{std::span<std::byte>{slab}}, n{load(path)},
        model{std::span<const fe::TensorView>{views.data(), n}, arena}
  {
  }

  std::size_t load(const char* path) noexcept
  {
    std::size_t count{0uz};
    if (!fe::load_safetensors(path, arena, std::span{views}, count))
      return 0uz;
    return count;
  }
};

fe_engine* fe_engine_load(const char* path)
{
  try {
    auto engine = std::make_unique<FeEngine>(path);
    return engine->model.valid() ? engine.release() : nullptr;
  } catch (const std::exception&) {
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

int fe_engine_run(fe_engine* engine, const std::int32_t* tokens, std::size_t seq_len, float* out)
{
  if (engine == nullptr || tokens == nullptr || out == nullptr)
    return 1;

  fe::Arena& arena = engine->arena;
  std::byte* const mark = arena.mark();
  const fe::MambaConfig& c = engine->model.config();
  const std::size_t hz = seq_len * c.d_model;

  auto* const in = arena.alloc_array<float>(hz, fe::kSimdAlign);
  if (in == nullptr) {
    arena.reset_to(mark);
    return 2;
  }
  const float* const emb = engine->model.embedding();
  for (std::size_t t{0uz}; t < seq_len; ++t) {
    const std::int32_t tok = tokens[t];
    if (std::cmp_less(tok, 0) || std::cmp_greater_equal(tok, c.vocab)) {
      arena.reset_to(mark);
      return 3;
    }
    const auto utok = static_cast<std::size_t>(tok);
    for (std::size_t i{0uz}; i < c.d_model; ++i)
      in[(t * c.d_model) + i] = emb[(utok * c.d_model) + i];
  }

  engine->model.forward({in, hz}, {out, hz}, seq_len);
  arena.reset_to(mark);
  return 0;
}