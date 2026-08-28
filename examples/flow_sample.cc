// Usage: flow_sample <mamba_flow.safetensors> [euler|heun|rk4] [steps]
#include "api/engine.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

int main(int argc, char** argv)
{
  fe_engine* engine = fe_engine_load((argc > 1) ? argv[1] : "");
  if (engine == nullptr) {
    std::cerr << "error: cannot load model\n";
    return 1;
  }
  const std::size_t a = fe_engine_action_dim(engine);
  if (a == 0uz) {
    std::cerr << "error: checkpoint has no flow head\n";
    fe_engine_free(engine);
    return 1;
  }
  const std::string_view m_arg = (argc > 2) ? argv[2] : "";
  const int method = (m_arg.empty() || m_arg == "euler") ? FE_SOLVER_EULER
                     : (m_arg == "heun")                 ? FE_SOLVER_HEUN
                     : (m_arg == "rk4")                  ? FE_SOLVER_RK4
                                                         : -1;
  if (method < 0) {
    std::cerr << "error: solver must be euler, heun, or rk4\n";
    fe_engine_free(engine);
    return 1;
  }
  const std::size_t steps = (argc > 3) ? std::stoull(argv[3]) : 10uz;

  const std::array<std::int32_t, 4> prefix{1, 2, 3, 4};
  std::vector<float> noise(a);
  std::vector<float> action(a);
  for (std::size_t i{0uz}; i < a; ++i)
    noise[i] = std::sin(static_cast<float>(i) * 0.3F);

  const int rc = fe_engine_sample(engine, prefix.data(), prefix.size(), noise.data(), steps, method,
                                  action.data());
  const char* solver = (method == FE_SOLVER_RK4)    ? "rk4"
                       : (method == FE_SOLVER_HEUN) ? "heun"
                                                    : "euler";
  if (rc == 0) {
    std::cout << "action_dim=" << a << "  solver=" << solver << "  NFE=" << steps
              << "  action[0..2]=" << action[0] << ", " << action[1 % a] << ", " << action[2 % a]
              << '\n';
  } else {
    std::cerr << "sample failed: " << fe_engine_last_error() << " (rc=" << rc << ")\n";
  }
  fe_engine_free(engine);
  return rc;
}
