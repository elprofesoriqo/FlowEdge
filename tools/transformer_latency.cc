#include "api/engine.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

namespace {

[[nodiscard]] bool parse_u32(std::string_view text, unsigned& value)
{
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc{} && end == text.data() + text.size();
}

[[nodiscard]] double percentile_ms(std::vector<double>& samples, double fraction)
{
  if (samples.empty())
    return 0.0;
  std::sort(samples.begin(), samples.end());
  const double index = fraction * static_cast<double>(samples.size() - 1uz);
  const auto lo = static_cast<std::size_t>(index);
  const auto hi = std::min(samples.size() - 1uz, lo + 1uz);
  const double w = index - static_cast<double>(lo);
  return samples[lo] * (1.0 - w) + samples[hi] * w;
}

void json_string(std::ostream& out, std::string_view text)
{
  out << '"';
  for (const char ch : text) {
    if (ch == '\\' || ch == '"')
      out << '\\';
    out << ch;
  }
  out << '"';
}

} // namespace

int main(int argc, char** argv)
{
  if (argc == 2 && std::string_view{argv[1]} == "--help") {
    std::cout
        << "usage: " << argv[0]
        << " MODEL [--threads N] [--warmup N] [--iters N] [--output PATH] TOKEN [TOKEN ...]\n";
    return 0;
  }
  if (argc < 3) {
    std::cerr
        << "usage: " << argv[0]
        << " MODEL [--threads N] [--warmup N] [--iters N] [--output PATH] TOKEN [TOKEN ...]\n";
    return 2;
  }

  unsigned threads{0u};
  unsigned warmup{5u};
  unsigned iters{20u};
  const char* output_path{};
  std::vector<std::int32_t> tokens;
  for (int index{2}; index < argc; ++index) {
    const std::string_view input{argv[index]};
    if (input == "--threads" && index + 1 < argc) {
      if (!parse_u32(argv[++index], threads)) {
        std::cerr << "error: --threads must be an unsigned integer\n";
        return 2;
      }
      continue;
    }
    if (input == "--warmup" && index + 1 < argc) {
      if (!parse_u32(argv[++index], warmup)) {
        std::cerr << "error: --warmup must be an unsigned integer\n";
        return 2;
      }
      continue;
    }
    if (input == "--iters" && index + 1 < argc) {
      if (!parse_u32(argv[++index], iters) || iters == 0u) {
        std::cerr << "error: --iters must be a positive integer\n";
        return 2;
      }
      continue;
    }
    if (input == "--output" && index + 1 < argc) {
      output_path = argv[++index];
      continue;
    }
    std::int32_t token{};
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

  const std::unique_ptr<fe_engine, decltype(&fe_engine_free)> engine{
      fe_engine_load_with_threads(argv[1], threads), fe_engine_free};
  if (!engine) {
    std::cerr << "error: cannot load model: " << fe_engine_last_error() << '\n';
    return 1;
  }
  std::size_t d_model{};
  std::size_t n_layers{};
  fe_engine_dims(engine.get(), &d_model, &n_layers);
  std::vector<float> out(tokens.size() * d_model);
  const auto time_prefill = [&]() -> double {
    const auto start = std::chrono::steady_clock::now();
    if (fe_engine_run(engine.get(), tokens.data(), tokens.size(), out.data()) != 0)
      return -1.0;
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
  };
  const auto time_stream = [&]() -> double {
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t index{}; index < tokens.size(); ++index) {
      if (fe_engine_step(engine.get(), tokens[index], out.data() + (index * d_model)) != 0)
        return -1.0;
    }
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
  };

  for (unsigned i{}; i < warmup; ++i) {
    if (time_prefill() < 0.0 || time_stream() < 0.0) {
      std::cerr << "error: warmup failed: " << fe_engine_last_error() << '\n';
      return 1;
    }
  }
  std::vector<double> prefill;
  std::vector<double> stream;
  prefill.reserve(iters);
  stream.reserve(iters);
  for (unsigned i{}; i < iters; ++i) {
    const double prefill_ms = time_prefill();
    const double stream_ms = time_stream();
    if (prefill_ms < 0.0 || stream_ms < 0.0) {
      std::cerr << "error: timed run failed: " << fe_engine_last_error() << '\n';
      return 1;
    }
    prefill.push_back(prefill_ms);
    stream.push_back(stream_ms);
  }

  const double prefill_p50 = percentile_ms(prefill, 0.50);
  const double prefill_p99 = percentile_ms(prefill, 0.99);
  const double stream_p50 = percentile_ms(stream, 0.50);
  const double stream_p99 = percentile_ms(stream, 0.99);
  auto write = [&](std::ostream& out_stream) {
    out_stream << "{\n  \"schema_version\": 1,\n  \"model\": ";
    json_string(out_stream, argv[1]);
    out_stream << ",\n  \"threads\": " << threads << ",\n  \"warmup\": " << warmup
               << ",\n  \"iters\": " << iters << ",\n  \"d_model\": " << d_model
               << ",\n  \"n_layers\": " << n_layers << ",\n  \"prefix\": [";
    for (std::size_t index{}; index < tokens.size(); ++index) {
      if (index != 0uz)
        out_stream << ", ";
      out_stream << tokens[index];
    }
    out_stream
        << "],\n  \"prefill_p50_ms\": " << prefill_p50 << ",\n  \"prefill_p99_ms\": " << prefill_p99
        << ",\n  \"stream_p50_ms\": " << stream_p50 << ",\n  \"stream_p99_ms\": " << stream_p99
        << ",\n  \"scope\": \"GPT-2-style decoder latency on this host; not policy inference\"\n"
        << "}\n";
  };
  write(std::cout);
  if (output_path != nullptr) {
    std::ofstream file{output_path};
    if (!file) {
      std::cerr << "error: cannot write " << output_path << '\n';
      return 1;
    }
    write(file);
  }
  return 0;
}
