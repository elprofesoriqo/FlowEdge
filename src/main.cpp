#include "../api/engine.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

int main(int argc, char** argv)
{
  const char* path = (argc > 1) ? argv[1] : "";
  fe_engine* engine = fe_engine_load(path);
  if (engine == nullptr) {
    std::cerr << "error: cannot load model\n";
    return 1;
  }

  std::size_t d_model{0uz};
  std::size_t n_layers{0uz};
  fe_engine_dims(engine, &d_model, &n_layers);

  const std::array<std::int32_t, 4> tokens{1, 2, 3, 4};
  std::vector<float> out(tokens.size() * d_model);
  const int rc = fe_engine_run(engine, tokens.data(), tokens.size(), out.data());

  if (rc == 0) {
    double ss{0.0};
    for (const float v : out)
      ss += static_cast<double>(v) * v;
    std::cout << "Mamba: " << n_layers << " layers, d_model=" << d_model
              << ", seq=" << tokens.size() << " -> ||h||=" << std::sqrt(ss) << '\n';
  } else {
    std::cerr << "error: forward failed\n";
  }

  fe_engine_free(engine);
  return rc;
}
