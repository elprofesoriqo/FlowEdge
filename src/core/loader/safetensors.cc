#include "safetensors.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstring>
#include <expected>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0602
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace fe {
namespace {

struct FileCloser
{
  void operator()(FILE* fp) const noexcept
  {
    std::fclose(fp); // NOLINT(cppcoreguidelines-owning-memory)
  }
};

struct MappedFile
{
  const std::byte* data{};
  std::size_t size{0uz};

#ifdef _WIN32
  HANDLE file{INVALID_HANDLE_VALUE}, mapping{};

  [[nodiscard]] static std::expected<MappedFile, const char*> open(const char* p) noexcept
  {
    MappedFile mf;
    mf.file = CreateFileA(p, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                          FILE_ATTRIBUTE_NORMAL, nullptr);
    if (mf.file == INVALID_HANDLE_VALUE)
      return std::unexpected("Failed to open safetensors file");
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(mf.file, &sz)) {
      mf.close();
      return std::unexpected("Failed to get safetensors file size");
    }
    mf.size = static_cast<std::size_t>(sz.QuadPart);
    mf.mapping = CreateFileMappingA(mf.file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mf.mapping) {
      mf.close();
      return std::unexpected("Failed to create file mapping");
    }
    mf.data = static_cast<const std::byte*>(MapViewOfFile(mf.mapping, FILE_MAP_READ, 0, 0, 0));
    if (!mf.data) {
      mf.close();
      return std::unexpected("Failed to map view of file");
    }
#if _WIN32_WINNT >= 0x0602
    WIN32_MEMORY_RANGE_ENTRY range{const_cast<void*>(static_cast<const void*>(mf.data)), mf.size};
    PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
#endif
    return mf;
  }
  void close() noexcept
  {
    if (data)
      UnmapViewOfFile(data);
    if (mapping)
      CloseHandle(mapping);
    if (file != INVALID_HANDLE_VALUE)
      CloseHandle(file);
    data = nullptr;
    mapping = nullptr;
    file = INVALID_HANDLE_VALUE;
  }
#else
  void* mapped_data{};

  [[nodiscard]] static std::expected<MappedFile, const char*> open(const char* p) noexcept
  {
    std::unique_ptr<FILE, FileCloser> fp{std::fopen(p, "rb")};
    if (!fp)
      return std::unexpected("Failed to open safetensors file");
    const int fd = fileno(fp.get());
    struct stat st = {};
    if (fstat(fd, &st) < 0)
      return std::unexpected("Failed to stat safetensors file");
    MappedFile mf;
    mf.size = static_cast<std::size_t>(st.st_size);
    mf.mapped_data = mmap(nullptr, mf.size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mf.mapped_data == MAP_FAILED)
      return std::unexpected("Failed to mmap safetensors file");
    mf.data = static_cast<const std::byte*>(mf.mapped_data);
#if defined(__linux__)
    madvise(mf.mapped_data, mf.size, MADV_SEQUENTIAL | MADV_WILLNEED);
#endif
    return mf;
  }
  void close() noexcept
  {
    if (mapped_data)
      munmap(mapped_data, size);
    mapped_data = nullptr;
    data = nullptr;
  }
#endif
};

[[nodiscard]] std::string_view after_colon(std::string_view json, std::string_view key) noexcept
{
  auto pos = json.find(key);
  if (pos == std::string_view::npos)
    return {};
  pos = json.find(':', pos + key.size());
  if (pos == std::string_view::npos)
    return {};
  return json.substr(pos + 1);
}

[[nodiscard]] std::size_t parse_u64s(std::string_view s, std::span<std::uint64_t> out) noexcept
{
  std::size_t n{0uz};
  for (auto p = s.data(), end = p + s.size(); p < end && n < out.size();) {
    while (p < end && (*p < '0' || *p > '9') && *p != ']')
      ++p;
    if (p >= end || *p == ']')
      break;
    std::uint64_t v{};
    while (p < end && *p >= '0' && *p <= '9')
      v = v * 10u + static_cast<std::uint64_t>(*p++ - '0');
    out[n++] = v;
  }
  return n;
}

template<typename Cb> [[nodiscard]] bool foreach_tensor(std::string_view json, Cb cb) noexcept
{
  const char* p = json.data();
  const char* end = p + json.size();
  while (p < end && *p != '{')
    ++p;
  if (p >= end)
    return false;
  ++p;

  while (p < end) {
    while (p < end && *p != '"' && *p != '}')
      ++p;
    if (p >= end || *p == '}')
      break;
    ++p;

    const auto* ns = p;
    while (p < end && *p != '"')
      ++p;
    std::string_view name{ns, static_cast<std::size_t>(p - ns)};
    if (p < end)
      ++p;

    while (p < end && *p != '{' && *p != '}')
      ++p;
    if (p >= end || *p == '}')
      break;

    int depth{1};
    const auto* os = p++;
    while (p < end && depth) {
      depth += (*p == '{') - (*p == '}');
      ++p;
    }
    std::string_view obj{os, static_cast<std::size_t>(p - os)};

    if (name == "__metadata__")
      continue;
    const bool bf16 = obj.find("\"BF16\"") != std::string_view::npos;
    if (!bf16 && obj.find("\"F32\"") == std::string_view::npos)
      continue; // only F32 and BF16

    std::array<std::uint64_t, 4> shape{};
    auto ndim = parse_u64s(after_colon(obj, "\"shape\""), std::span{shape});
    std::array<std::uint64_t, 2> offs{};
    if (parse_u64s(after_colon(obj, "\"data_offsets\""), std::span{offs}) < 2)
      continue;

    if (!cb(name, offs[0], offs[1] - offs[0], shape, static_cast<std::uint8_t>(ndim), bf16))
      return false;
  }
  return true;
}

struct MappedJson
{
  MappedFile mf;
  std::string_view json;
};

[[nodiscard]] std::size_t quoted_key_pos(std::string_view json, std::string_view key) noexcept
{
  std::size_t position = json.find(key);
  while (position != std::string_view::npos) {
    if (position > 0uz && position + key.size() < json.size() && json[position - 1uz] == '"' &&
        json[position + key.size()] == '"')
      return position - 1uz;
    position = json.find(key, position + 1uz);
  }
  return std::string_view::npos;
}

[[nodiscard]] std::string_view object_value(std::string_view json, std::string_view key) noexcept
{
  const std::size_t key_pos = quoted_key_pos(json, key);
  if (key_pos == std::string_view::npos)
    return {};
  const std::size_t colon = json.find(':', key_pos + key.size() + 2uz);
  if (colon == std::string_view::npos)
    return {};
  const std::size_t start = json.find('{', colon + 1uz);
  if (start == std::string_view::npos)
    return {};

  int depth{0};
  bool in_string{false};
  for (std::size_t index{start}; index < json.size(); ++index) {
    const char ch = json[index];
    if (ch == '"' && (index == 0uz || json[index - 1uz] != '\\'))
      in_string = !in_string;
    if (in_string)
      continue;
    if (ch == '{')
      ++depth;
    else if (ch == '}' && --depth == 0)
      return json.substr(start, index - start + 1uz);
  }
  return {};
}

struct MetadataString
{
  std::string_view value{};
  bool present{false};
  bool valid{false};
};

[[nodiscard]] MetadataString metadata_string_value(std::string_view object,
                                                   std::string_view key) noexcept
{
  const std::size_t key_pos = quoted_key_pos(object, key);
  if (key_pos == std::string_view::npos)
    return {};
  const std::size_t colon = object.find(':', key_pos + key.size() + 2uz);
  if (colon == std::string_view::npos)
    return {.present = true};
  std::size_t start = colon + 1uz;
  while (start < object.size() && (object[start] == ' ' || object[start] == '\n' ||
                                   object[start] == '\r' || object[start] == '\t'))
    ++start;
  if (start >= object.size() || object[start] != '"')
    return {.present = true};
  for (std::size_t end{start + 1uz}; end < object.size(); ++end) {
    if (object[end] == '"' && object[end - 1uz] != '\\')
      return {.value = object.substr(start + 1uz, end - start - 1uz),
              .present = true,
              .valid = true};
  }
  return {.present = true};
}

[[nodiscard]] std::expected<std::string, const char*> unescape_json_string(
    std::string_view escaped) noexcept
{
  try {
    std::string value;
    value.reserve(escaped.size());
    for (std::size_t index{}; index < escaped.size(); ++index) {
      if (escaped[index] != '\\') {
        value.push_back(escaped[index]);
        continue;
      }
      if (++index >= escaped.size())
        return std::unexpected("Invalid escaped metadata string");
      switch (escaped[index]) {
      case '"':
        value.push_back('"');
        break;
      case '\\':
        value.push_back('\\');
        break;
      case '/':
        value.push_back('/');
        break;
      case 'b':
        value.push_back('\b');
        break;
      case 'f':
        value.push_back('\f');
        break;
      case 'n':
        value.push_back('\n');
        break;
      case 'r':
        value.push_back('\r');
        break;
      case 't':
        value.push_back('\t');
        break;
      default:
        return std::unexpected("Unsupported escaped metadata string");
      }
    }
    return value;
  } catch (...) {
    return std::unexpected("Out of memory reading safetensors metadata");
  }
}

[[nodiscard]] std::expected<MappedJson, const char*> header_json(std::string_view path) noexcept
{
  std::array<char, 512> buf{};
  if (path.size() >= buf.size())
    return std::unexpected("Path too long");
  std::memcpy(buf.data(), path.data(), path.size());

  auto mf_res = MappedFile::open(buf.data());
  if (!mf_res)
    return std::unexpected(mf_res.error());

  MappedFile mf = *mf_res;
  if (mf.size < 8uz) {
    mf.close();
    return std::unexpected("File too small");
  }
  std::uint64_t header_len{};
  std::memcpy(&header_len, mf.data, 8uz);
  if constexpr (std::endian::native == std::endian::big)
    header_len = std::byteswap(header_len); // safetensors encodes its length as little-endian
  if (header_len > mf.size - 8uz) {         // overflow-safe: mf.size >= 8
    mf.close();
    return std::unexpected("Invalid header length");
  }
  return MappedJson{.mf = mf,
                    .json = {reinterpret_cast<const char*>(mf.data + 8uz),
                             static_cast<std::size_t>(header_len)}};
}

} // namespace

std::expected<void, const char*> load_safetensors(std::string_view path, Arena& arena,
                                                  std::span<TensorView> out,
                                                  std::size_t& tensors_loaded) noexcept
{
  tensors_loaded = 0uz;
  auto res = header_json(path);
  if (!res)
    return std::unexpected(res.error());

  MappedJson mapped = *res;
  MappedFile mf = mapped.mf;
  const std::string_view json = mapped.json;
  const std::byte* const weights_base = mf.data + 8uz + json.size();
  const std::uint64_t data_size = mf.size - 8uz - json.size();

  const bool ok =
      foreach_tensor(json,
                     [&](std::string_view name, std::uint64_t byte_off, std::uint64_t byte_len,
                         const std::array<std::uint64_t, 4>& shape, std::uint8_t ndim,
                         bool bf16) noexcept -> bool {
                       if (tensors_loaded >= out.size()) [[unlikely]]
                         return false;
                       if (byte_len > data_size || byte_off > data_size - byte_len)
                           [[unlikely]] // in-bounds
                         return false;
                       const std::size_t elem = bf16 ? 2uz : sizeof(float);
                       if (byte_len % elem != 0uz) [[unlikely]] // whole elements
                         return false;

                       // store bytes; matmul widens inline
                       auto* const dst = arena.alloc_array<std::byte, kSimdAlign>(byte_len);
                       if (!dst) [[unlikely]]
                         return false;
                       std::memcpy(dst, weights_base + byte_off, byte_len);

                       TensorView& tv = out[tensors_loaded++];
                       tv.data = dst;
                       tv.bytes = static_cast<std::size_t>(byte_len);
                       tv.dtype = bf16 ? TensorView::Dtype::BF16 : TensorView::Dtype::F32;
                       tv.ndim = ndim;
                       for (std::size_t i{0uz}; i < 4uz; ++i)
                         tv.shape[i] = (i < ndim) ? static_cast<std::size_t>(shape[i]) : 0uz;
                       const std::size_t nlen = std::min(name.size(), tv.name.size() - 1uz);
                       std::memcpy(tv.name.data(), name.data(), nlen);
                       tv.name[nlen] = '\0';
                       return true;
                     });

  mf.close();
  if (!ok)
    return std::unexpected("Failed during tensor iteration or out of bounds");
  return {};
}

std::size_t safetensors_weight_bytes(std::string_view path) noexcept
{
  auto res = header_json(path);
  if (!res)
    return 0uz;

  MappedJson mapped = *res;
  MappedFile mf = mapped.mf;
  const std::string_view json = mapped.json;
  std::size_t total{0uz};
  static_cast<void>(
      foreach_tensor(json, [&](std::string_view, std::uint64_t, std::uint64_t byte_len,
                               const std::array<std::uint64_t, 4>&, std::uint8_t, bool) noexcept {
        total += byte_len; // BF16 stays 2 bytes/elem
        return true;
      }));
  mf.close();
  return total;
}

std::array<std::size_t, 4> safetensors_tensor_shape(std::string_view path,
                                                    std::string_view name) noexcept
{
  auto res = header_json(path);
  if (!res)
    return {};

  MappedJson mapped = *res;
  MappedFile mf = mapped.mf;
  const std::string_view json = mapped.json;
  std::array<std::size_t, 4> out{};
  static_cast<void>(foreach_tensor(json, [&](std::string_view n, std::uint64_t, std::uint64_t,
                                             const std::array<std::uint64_t, 4>& shape,
                                             std::uint8_t ndim, bool) noexcept {
    if (n != name)
      return true; // keep scanning
    for (std::size_t i{0uz}; i < ndim; ++i)
      out[i] = static_cast<std::size_t>(shape[i]);
    return false; // found
  }));
  mf.close();
  return out;
}

std::expected<std::optional<std::string>, const char*> safetensors_metadata_value(
    std::string_view path, std::string_view key) noexcept
{
  auto res = header_json(path);
  if (!res)
    return std::unexpected(res.error());

  MappedJson mapped = *res;
  const std::string_view metadata = object_value(mapped.json, "__metadata__");
  if (metadata.empty()) {
    mapped.mf.close();
    return std::optional<std::string>{};
  }
  const MetadataString encoded = metadata_string_value(metadata, key);
  if (!encoded.present) {
    mapped.mf.close();
    return std::optional<std::string>{};
  }
  if (!encoded.valid) {
    mapped.mf.close();
    return std::unexpected("Invalid safetensors metadata value");
  }
  const auto decoded = unescape_json_string(encoded.value);
  mapped.mf.close();
  if (!decoded)
    return std::unexpected(decoded.error());
  return std::optional<std::string>{std::move(*decoded)};
}

ModelWeights::ModelWeights(std::size_t weight_bytes,
                           std::optional<DeploymentProfile> deployment_profile)
    : storage_(weight_bytes + (kMaxTensors * (kSimdAlign - 1uz))),
      arena_{std::span<std::byte>{storage_.data(), storage_.size()}}, weight_bytes_{weight_bytes}
{
  deployment_profile_ = std::move(deployment_profile);
}

std::expected<std::shared_ptr<const ModelWeights>, const char*> ModelWeights::open(
    std::string_view path) noexcept
{
  std::optional<DeploymentProfile> deployment_profile;
  try {
    const auto encoded_profile = safetensors_metadata_value(path, kDeploymentProfileMetadataKey);
    if (!encoded_profile)
      return std::unexpected(encoded_profile.error());
    if (*encoded_profile) {
      const auto parsed_profile = parse_deployment_profile(**encoded_profile);
      if (!parsed_profile)
        return std::unexpected(deployment_profile_error_message(parsed_profile.error()).data());
      deployment_profile = std::move(*parsed_profile);
    }
  } catch (...) {
    return std::unexpected("Out of memory parsing deployment profile");
  }
  const std::size_t bytes = safetensors_weight_bytes(path);
  if (bytes == 0uz)
    return std::unexpected("Failed to load safetensors file or find supported tensors");
  constexpr std::size_t alignment_padding = kMaxTensors * (kSimdAlign - 1uz);
  if (bytes > std::numeric_limits<std::size_t>::max() - alignment_padding)
    return std::unexpected("Checkpoint weight storage size overflows this platform");
  try {
    std::shared_ptr<ModelWeights> weights{new ModelWeights{bytes, std::move(deployment_profile)}};
    const auto loaded =
        load_safetensors(path, weights->arena_, weights->views_, weights->tensor_count_);
    if (!loaded)
      return std::unexpected(loaded.error());
    return std::shared_ptr<const ModelWeights>{std::move(weights)};
  } catch (...) {
    return std::unexpected("Out of memory loading immutable model weights");
  }
}

} // namespace fe
