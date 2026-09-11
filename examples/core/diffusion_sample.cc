#include "api/engine.h"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

namespace {

struct EngineDeleter
{
  void operator()(fe_engine* engine) const noexcept { fe_engine_free(engine); }
};

} // namespace

int main(int argc, char** argv)
{
  if (argc < 2) {
    std::cerr << "usage: diffusion_sample <converted.safetensors> [steps]\n";
    return 1;
  }
  const std::size_t steps = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 10uz;
  const std::unique_ptr<fe_engine, EngineDeleter> engine{fe_engine_load(argv[1])};
  if (!engine) {
    std::cerr << "load failed: " << fe_engine_last_error() << '\n';
    return 2;
  }
  const std::size_t condition_dim = fe_engine_condition_dim(engine.get());
  const std::size_t action_dim = fe_engine_action_dim(engine.get());
  const std::size_t horizon = fe_engine_action_horizon(engine.get());
  if (condition_dim == 0uz || action_dim == 0uz || horizon <= 1uz) {
    std::cerr << "checkpoint does not contain a Diffusion Policy head\n";
    return 3;
  }

  // In an application this comes from the policy's observation encoder. The
  // reference PushT checkpoint expects 132 values (two observation features).
  std::vector<float> condition(condition_dim, 0.0F);
  std::vector<float> noise(horizon * action_dim);
  for (std::size_t i{0uz}; i < noise.size(); ++i)
    noise[i] = std::sin(0.37F * static_cast<float>(i + 1uz));
  std::vector<float> action(noise.size());
  if (fe_engine_sample_diffusion(engine.get(), condition.data(), noise.data(), steps,
                                 FE_DIFFUSION_DDIM, 0u, action.data()) != 0) {
    std::cerr << "sampling failed: " << fe_engine_last_error() << '\n';
    return 4;
  }

  std::cout << "action_horizon=" << horizon << " action_dim=" << action_dim << '\n';
  for (std::size_t t{0uz}; t < horizon; ++t) {
    std::cout << t << ':';
    for (std::size_t d{0uz}; d < action_dim; ++d)
      std::cout << ' ' << action[(t * action_dim) + d];
    std::cout << '\n';
  }
  return 0;
}
