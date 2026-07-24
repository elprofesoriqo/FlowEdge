// Usage: flow_sample <mamba_flow.safetensors> [euler|heun] [steps]
#include "api/engine.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string_view>

int main(int argc, char** argv)
{
  fe_engine* engine = fe_engine_load((argc > 1) ? argv[1] : "");
  if (engine == nullptr) {
    std::fprintf(stderr, "error: cannot load model\n");
    return 1;
  }
  const std::size_t a = fe_engine_action_dim(engine);
  if (a == 0uz) {
    std::fprintf(stderr, "error: checkpoint has no flow head\n");
    fe_engine_free(engine);
    return 1;
  }
  const int method = (argc > 2 && std::string_view{argv[2]} == "heun") ? 1 : 0;
  const std::size_t steps = (argc > 3) ? std::strtoul(argv[3], nullptr, 10) : 10uz;

  const std::array<std::int32_t, 4> prefix{1, 2, 3, 4};
  auto noise = std::make_unique<float[]>(a);
  auto action = std::make_unique<float[]>(a);
  for (std::size_t i{0uz}; i < a; ++i)
    noise[i] = std::sin(static_cast<float>(i) * 0.3F);

  const int rc = fe_engine_sample(engine, prefix.data(), prefix.size(), noise.get(), steps, method,
                                  action.get());
  if (rc == 0) {
    std::printf("action_dim=%zu  solver=%s  NFE=%zu  action[0..2]=%f, %f, %f\n",
                a, method ? "heun" : "euler", steps, action[0], action[1 % a], action[2 % a]);
  } else {
    std::fprintf(stderr, "sample failed rc=%d\n", rc);
  }
  fe_engine_free(engine);
  return rc;
}
