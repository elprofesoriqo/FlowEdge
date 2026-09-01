#include "api/engine.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cerr << "usage: external_flow_sample <model.safetensors>\n";
    return 2;
  }

  fe_engine* const engine = fe_engine_load(argv[1]);
  if (engine == nullptr) {
    std::cerr << "load failed: " << fe_engine_last_error() << '\n';
    return 1;
  }

  const std::size_t condition_dim = fe_engine_condition_dim(engine);
  const std::size_t action_dim = fe_engine_action_dim(engine);
  std::vector<float> condition(condition_dim), noise(action_dim), direct(action_dim),
      resumed(action_dim);
  for (std::size_t i{0uz}; i < condition.size(); ++i)
    condition[i] = -0.1F + (0.2F * static_cast<float>(i) / static_cast<float>(condition.size()));
  for (std::size_t i{0uz}; i < noise.size(); ++i)
    noise[i] = -0.2F + (0.4F * static_cast<float>(i) / static_cast<float>(noise.size()));

  constexpr std::size_t k_steps = 6uz;
  int rc = fe_engine_sample_condition(engine, condition.data(), noise.data(), k_steps,
                                      FE_SOLVER_HEUN, direct.data());
  if (rc == 0)
    rc = fe_engine_flow_begin(engine, condition.data(), noise.data(), k_steps, FE_SOLVER_HEUN);
  std::size_t remaining{k_steps};
  if (rc == 0)
    rc = fe_engine_flow_advance(engine, 2uz, resumed.data(), &remaining);
  if (rc == 0)
    rc = fe_engine_flow_advance(engine, k_steps, resumed.data(), &remaining);

  const bool exact = rc == 0 && remaining == 0uz &&
                     std::ranges::equal(direct, resumed, [](float a, float b) { return a == b; });
  std::cout << "condition_dim=" << condition_dim << " action_dim=" << action_dim
            << " resumable_exact=" << (exact ? "true" : "false") << '\n';
  if (rc != 0)
    std::cerr << "sample failed: " << fe_engine_last_error() << '\n';
  fe_engine_free(engine);
  return exact ? 0 : 1;
}
