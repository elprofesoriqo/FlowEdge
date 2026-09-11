#include "api/engine.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cerr << "usage: external_flow_sample <model.safetensors>\n";
    return 2;
  }

  const std::unique_ptr<fe_engine, decltype(&fe_engine_free)> engine{fe_engine_load(argv[1]),
                                                                     fe_engine_free};
  if (!engine) {
    std::cerr << "load failed: " << fe_engine_last_error() << '\n';
    return 1;
  }

  const std::size_t condition_dim = fe_engine_condition_dim(engine.get());
  const std::size_t action_dim = fe_engine_action_dim(engine.get());
  std::vector<float> condition(condition_dim), noise(action_dim), direct(action_dim),
      resumed(action_dim);
  for (std::size_t i{0uz}; i < condition.size(); ++i)
    condition[i] = -0.1F + (0.2F * static_cast<float>(i) / static_cast<float>(condition.size()));
  for (std::size_t i{0uz}; i < noise.size(); ++i)
    noise[i] = -0.2F + (0.4F * static_cast<float>(i) / static_cast<float>(noise.size()));

  constexpr std::size_t k_steps = 6uz;
  int rc = fe_engine_sample_condition(engine.get(), condition.data(), noise.data(), k_steps,
                                      FE_SOLVER_HEUN, direct.data());
  fe_condition_metadata request{};
  if (rc == 0)
    rc = fe_engine_make_condition_metadata(engine.get(), 1u, 2u, 7u, k_steps, FE_SOLVER_HEUN,
                                           &request);
  if (rc == 0)
    rc = fe_engine_flow_begin_request(engine.get(), condition.data(), noise.data(), &request);
  std::size_t remaining{k_steps};
  if (rc == 0)
    rc = fe_engine_flow_advance(engine.get(), 2uz, resumed.data(), &remaining);
  if (rc == 0)
    rc = fe_engine_flow_advance(engine.get(), k_steps, resumed.data(), &remaining);
  fe_action_metadata result{};
  if (rc == 0)
    rc = fe_engine_flow_action_metadata(engine.get(), &result);

  const bool exact = rc == 0 && remaining == 0uz && result.status == FE_ACTION_COMPLETE &&
                     result.remaining_nfe == 0u &&
                     std::ranges::equal(direct, resumed, [](float a, float b) { return a == b; });
  std::cout << "condition_dim=" << condition_dim << " action_dim=" << action_dim
            << " generation=" << result.generation
            << " resumable_exact=" << (exact ? "true" : "false") << '\n';
  if (rc != 0)
    std::cerr << "sample failed: " << fe_engine_last_error() << '\n';
  return exact ? 0 : 1;
}
