#include "api/engine.h"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

int main(int argc, char** argv)
{
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " MODEL TOKEN [TOKEN ...]\n";
    return 2;
  }
  const std::unique_ptr<fe_engine, decltype(&fe_engine_free)> engine{fe_engine_load(argv[1]),
                                                                     fe_engine_free};
  if (!engine) {
    std::cerr << "error: cannot load model\n";
    return 1;
  }

  std::size_t d_model{0uz};
  std::size_t n_layers{0uz};
  fe_engine_dims(engine.get(), &d_model, &n_layers);

  std::vector<std::int32_t> tokens;
  tokens.reserve(static_cast<std::size_t>(argc - 2));
  for (int index{2}; index < argc; ++index) {
    std::int32_t token{0};
    const std::string_view input{argv[index]};
    const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), token);
    if (error != std::errc{} || end != input.data() + input.size()) {
      std::cerr << "error: token must be a signed 32-bit integer: " << input << '\n';
      return 2;
    }
    tokens.push_back(token);
  }
  std::vector<float> out(tokens.size() * d_model);

  const int rc = fe_engine_run(engine.get(), tokens.data(), tokens.size(), out.data());
  if (rc == 0) {
    std::cout << n_layers << " layers, d_model=" << d_model << ", out[0]=" << out[0] << '\n';
  }

  return rc;
}
