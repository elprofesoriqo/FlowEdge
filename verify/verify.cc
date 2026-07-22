#include "../api/engine.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char** argv)
{
  const char* model = (argc > 1) ? argv[1] : "models/mamba.safetensors";
  const char* baseline = (argc > 2) ? argv[2] : "models/baseline.bin";
  const std::array<std::int32_t, 4> tokens{1, 2, 3, 4};

  fe_engine* engine = fe_engine_load(model);
  if (engine == nullptr) {
    std::cerr << "error: cannot load " << model << '\n';
    return 2;
  }
  std::size_t d_model{0uz};
  std::size_t n_layers{0uz};
  fe_engine_dims(engine, &d_model, &n_layers);

  const std::size_t n = tokens.size() * d_model;
  std::vector<float> out(n);
  const int rc = fe_engine_run(engine, tokens.data(), tokens.size(), out.data());
  fe_engine_free(engine);
  if (rc != 0) {
    std::cerr << "error: forward failed\n";
    return 2;
  }

  std::vector<float> base(n);
  std::ifstream file(baseline, std::ios::binary);
  if (!file.read(reinterpret_cast<char*>(base.data()),
                 static_cast<std::streamsize>(n * sizeof(float)))) {
    std::cerr << "error: cannot read " << baseline << '\n';
    return 2;
  }

  double max_abs{0.0};
  double max_rel{0.0};
  for (std::size_t i{0uz}; i < n; ++i) {
    const double diff = std::fabs(static_cast<double>(out[i]) - static_cast<double>(base[i]));
    max_abs = std::max(max_abs, diff);
    max_rel = std::max(max_rel, diff / (std::fabs(static_cast<double>(base[i])) + 1e-6));
  }

  constexpr double tol = 2e-2; // fp32 accumulation over 24 layers + the ~1 ULP exp8 gate
  const bool pass = max_rel < tol;
  std::cout << "verify: " << n_layers << " layers, " << n << " values | max_abs=" << max_abs
            << " max_rel=" << max_rel << " (tol=" << tol << ") -> " << (pass ? "PASS" : "FAIL")
            << '\n';
  return pass ? 0 : 1;
}
