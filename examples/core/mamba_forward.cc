#include "api/engine.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

int main(int argc, char** argv)
{
  const std::unique_ptr<fe_engine, decltype(&fe_engine_free)> engine{fe_engine_load(
                                                                         (argc > 1) ? argv[1] : ""),
                                                                     fe_engine_free};
  if (!engine) {
    std::cerr << "error: cannot load model\n";
    return 1;
  }

  std::size_t d_model{0uz};
  std::size_t n_layers{0uz};
  fe_engine_dims(engine.get(), &d_model, &n_layers);

  const std::array<std::int32_t, 4> tokens{1, 2, 3, 4};
  std::vector<float> out(tokens.size() * d_model);

  const int rc = fe_engine_run(engine.get(), tokens.data(), tokens.size(), out.data());
  if (rc == 0) {
    std::cout << n_layers << " layers, d_model=" << d_model << ", out[0]=" << out[0] << '\n';
  }

  return rc;
}
