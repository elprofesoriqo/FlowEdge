#include "api/engine.h"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string_view>
#include <vector>

int main(int argc, char** argv)
{
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " MODEL [--stream] [--json] TOKEN [TOKEN ...]\n";
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
  bool stream{};
  bool json{};
  for (int index{2}; index < argc; ++index) {
    const std::string_view input{argv[index]};
    if (input == "--stream") {
      stream = true;
      continue;
    }
    if (input == "--json") {
      json = true;
      continue;
    }
    std::int32_t token{0};
    const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), token);
    if (error != std::errc{} || end != input.data() + input.size()) {
      std::cerr << "error: token must be a signed 32-bit integer: " << input << '\n';
      return 2;
    }
    tokens.push_back(token);
  }
  if (tokens.empty()) {
    std::cerr << "error: supply at least one token\n";
    return 2;
  }
  std::vector<float> out(tokens.size() * d_model);

  int rc{};
  if (stream) {
    for (std::size_t index{}; index < tokens.size(); ++index) {
      rc = fe_engine_step(engine.get(), tokens[index], out.data() + (index * d_model));
      if (rc != 0)
        break;
    }
  } else {
    rc = fe_engine_run(engine.get(), tokens.data(), tokens.size(), out.data());
  }
  if (rc == 0) {
    if (!json) {
      std::cout << n_layers << " layers, d_model=" << d_model << ", out[0]=" << out[0] << '\n';
    } else {
      std::cout << std::setprecision(std::numeric_limits<float>::max_digits10) << "{\"mode\":\""
                << (stream ? "stream" : "prefill") << "\",\"d_model\":" << d_model
                << ",\"outputs\":[";
      for (std::size_t index{}; index < out.size(); ++index) {
        if (index != 0uz)
          std::cout << ',';
        std::cout << out[index];
      }
      std::cout << "]}\n";
    }
  }

  return rc;
}
