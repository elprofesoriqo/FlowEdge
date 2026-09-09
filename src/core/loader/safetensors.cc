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
    if (sz.QuadPart < 0 ||
        static_cast<unsigned long long>(sz.QuadPart) > std::numeric_limits<std::size_t>::max()) {
      mf.close();
      return std::unexpected("Safetensors file size is invalid");
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
    if (st.st_size < 0 ||
        static_cast<unsigned long long>(st.st_size) > std::numeric_limits<std::size_t>::max())
      return std::unexpected("Safetensors file size is invalid");
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

[[nodiscard]] std::optional<std::size_t> parse_u64s(std::string_view s,
                                                    std::span<std::uint64_t> out) noexcept
{
  std::size_t count{0uz};
  for (auto p = s.data(), end = p + s.size(); p < end;) {
    while (p < end && (*p < '0' || *p > '9') && *p != ']')
      ++p;
    if (p >= end)
      return std::nullopt;
    if (*p == ']')
      return count;
    std::uint64_t v{};
    while (p < end && *p >= '0' && *p <= '9') {
      const auto digit = static_cast<std::uint64_t>(*p++ - '0');
      if (v > (std::numeric_limits<std::uint64_t>::max() - digit) / 10u)
        return std::nullopt;
      v = v * 10u + digit;
    }
    if (count < out.size())
      out[count] = v;
    ++count;
  }
  return std::nullopt;
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

  bool root_closed{};
  while (p < end) {
    while (p < end && *p != '"' && *p != '}')
      ++p;
    if (p >= end)
      break;
    if (*p == '}') {
      root_closed = true;
      break;
    }
    ++p;

    const auto* ns = p;
    while (p < end && *p != '"')
      ++p;
    if (p >= end)
      return false;
    std::string_view name{ns, static_cast<std::size_t>(p - ns)};
    if (p < end)
      ++p;

    while (p < end && *p != '{' && *p != '}')
      ++p;
    if (p >= end || *p == '}')
      return false;

    int depth{1};
    const auto* os = p++;
    while (p < end && depth) {
      depth += (*p == '{') - (*p == '}');
      ++p;
    }
    if (depth != 0)
      return false;
    std::string_view obj{os, static_cast<std::size_t>(p - os)};

    if (name == "__metadata__")
      continue;
    const bool bf16 = obj.find("\"BF16\"") != std::string_view::npos;
    const bool f32 = obj.find("\"F32\"") != std::string_view::npos;
    const TensorView::Dtype dtype = bf16  ? TensorView::Dtype::BF16
                                    : f32 ? TensorView::Dtype::F32
                                          : TensorView::Dtype::Unsupported;

    std::array<std::uint64_t, 4> shape{};
    const auto shape_count = parse_u64s(after_colon(obj, "\"shape\""), std::span{shape});
    if (!shape_count || *shape_count > shape.size())
      return false;
    std::array<std::uint64_t, 2> offs{};
    const auto offset_count = parse_u64s(after_colon(obj, "\"data_offsets\""), std::span{offs});
    if (!offset_count || *offset_count != offs.size() || offs[1] < offs[0])
      return false;

    if (!cb(name, offs[0], offs[1] - offs[0], shape, static_cast<std::uint8_t>(*shape_count), bf16))
      return false;
  }
  return root_closed;
}

struct MappedJson
{
  MappedFile mf;
  std::string_view json;
};

struct TensorSpanTracker
{
  static constexpr std::size_t kMaxSpans = 1024uz;
  std::uint64_t data_size{};
  std::array<std::pair<std::uint64_t, std::uint64_t>, kMaxSpans> spans{};
  std::size_t count{};

  [[nodiscard]] bool add(std::uint64_t offset, std::uint64_t length) noexcept
  {
    if (offset > data_size || length > data_size - offset || count == spans.size())
      return false;
    const std::uint64_t end = offset + length;
    for (const auto [existing_offset, existing_end] : std::span{spans}.first(count))
      if (offset < existing_end && existing_offset < end)
        return false;
    spans[count++] = {offset, end};
    return true;
  }
};

[[nodiscard]] bool valid_tensor_shape(const std::array<std::uint64_t, 4>& shape, std::uint8_t ndim,
                                      std::uint64_t byte_len, bool bf16) noexcept
{
  const std::size_t elem = bf16 ? 2uz : sizeof(float);
  std::size_t elements{1uz};
  for (std::size_t index{}; index < ndim; ++index) {
    if (shape[index] > std::numeric_limits<std::size_t>::max() ||
        (shape[index] != 0uz && elements > std::numeric_limits<std::size_t>::max() / shape[index]))
      return false;
    elements *= static_cast<std::size_t>(shape[index]);
  }
  return elements <= std::numeric_limits<std::size_t>::max() / elem && elements * elem == byte_len;
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
  TensorSpanTracker spans{.data_size = data_size};

  const bool ok =
      foreach_tensor(json,
                     [&](std::string_view name, std::uint64_t byte_off, std::uint64_t byte_len,
                         const std::array<std::uint64_t, 4>& shape, std::uint8_t ndim,
                         TensorView::Dtype dtype) noexcept -> bool {
                       if (dtype == TensorView::Dtype::Unsupported)
                         return true;
                       if (tensors_loaded >= out.size()) [[unlikely]]
                         return false;
                       if (!spans.add(byte_off, byte_len)) [[unlikely]]
                         return false;
                       if (!valid_tensor_shape(shape, ndim, byte_len, bf16))
                         return false;
                       std::size_t elements{1uz};
                       for (std::size_t axis{0uz}; axis < ndim; ++axis) {
                         if (shape[axis] > std::numeric_limits<std::size_t>::max() / elements)
                             [[unlikely]]
                           return false;
                         elements *= static_cast<std::size_t>(shape[axis]);
                       }
                       if (elements > std::numeric_limits<std::size_t>::max() / elem ||
                           elements * elem != byte_len) [[unlikely]]
                         return false;

                       // store bytes; matmul widens inline
                       auto* const dst = arena.alloc_array<std::byte, kSimdAlign>(byte_len);
                       if (!dst) [[unlikely]]
                         return false;
                       std::memcpy(dst, weights_base + byte_off, byte_len);

                       TensorView& tv = out[tensors_loaded++];
                       tv.data = dst;
                       tv.bytes = static_cast<std::size_t>(byte_len);
                       tv.dtype = dtype;
                       tv.ndim = ndim;
                       for (std::size_t i{0uz}; i < 4uz; ++i)
                         tv.shape[i] = (i < ndim) ? static_cast<std::size_t>(shape[i]) : 0uz;
                       const std::size_t nlen = std::min(name.size(), tv.name.size() - 1uz);
                       std::memcpy(tv.name.data(), name.data(), nlen);
                       tv.name[nlen] = '\0';
                       return true;
                     });

  mf.close();
  if (!ok || tensors_loaded == 0uz)
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
  const std::uint64_t data_size = mf.size - 8uz - json.size();
  TensorSpanTracker spans{.data_size = data_size};
  std::size_t total{0uz};
  const bool ok =
      foreach_tensor(json, [&](std::string_view, std::uint64_t byte_off, std::uint64_t byte_len,
                               const std::array<std::uint64_t, 4>& shape, std::uint8_t ndim,
                               bool bf16) noexcept {
        if (!spans.add(byte_off, byte_len) || !valid_tensor_shape(shape, ndim, byte_len, bf16) ||
            byte_len > std::numeric_limits<std::size_t>::max() - total)
          return false;
        total += static_cast<std::size_t>(byte_len); // BF16 stays 2 bytes/elem
        return true;
      });
  mf.close();
  return ok ? total : 0uz;
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
                                             std::uint8_t ndim, TensorView::Dtype) noexcept {
    if (n != name)
      return true; // keep scanning
    for (std::size_t i{0uz}; i < ndim; ++i)
      out[i] = static_cast<std::size_t>(shape[i]);
    return false; // found
  }));
  mf.close();
  return out;
}

std::expected<void, const char*> inspect_safetensors(std::string_view path,
                                                     std::span<TensorMetadata> out,
                                                     std::size_t& tensors_loaded,
                                                     std::size_t& total_bytes,
                                                     std::size_t& unsupported_tensors) noexcept
{
  tensors_loaded = 0uz;
  total_bytes = 0uz;
  unsupported_tensors = 0uz;
  auto res = header_json(path);
  if (!res)
    return std::unexpected(res.error());

  MappedJson mapped = *res;
  MappedFile mf = mapped.mf;
  const bool ok =
      foreach_tensor(mapped.json,
                     [&](std::string_view name, std::uint64_t, std::uint64_t byte_len,
                         const std::array<std::uint64_t, 4>& shape, std::uint8_t ndim,
                         TensorView::Dtype dtype) noexcept -> bool {
                       if (tensors_loaded >= out.size())
                         return false;
                       TensorMetadata& metadata = out[tensors_loaded++];
                       metadata.dtype = dtype;
                       metadata.bytes = static_cast<std::size_t>(byte_len);
                       metadata.ndim = ndim;
                       for (std::size_t i{0uz}; i < 4uz; ++i)
                         metadata.shape[i] = i < ndim ? static_cast<std::size_t>(shape[i]) : 0uz;
                       const std::size_t nlen = std::min(name.size(), metadata.name.size() - 1uz);
                       std::memcpy(metadata.name.data(), name.data(), nlen);
                       metadata.name[nlen] = '\0';
                       total_bytes += metadata.bytes;
                       unsupported_tensors += metadata.supported() ? 0uz : 1uz;
                       return true;
                     });
  mf.close();
  if (!ok)
    return std::unexpected("Failed during tensor inspection or output capacity was exceeded");
  return {};
}

ModelWeights::ModelWeights(std::size_t weight_bytes)
    : storage_(weight_bytes + (kMaxTensors * (kSimdAlign - 1uz))),
      arena_{std::span<std::byte>{storage_.data(), storage_.size()}}, weight_bytes_{weight_bytes}
{
}

std::expected<std::shared_ptr<const ModelWeights>, const char*> ModelWeights::open(
    std::string_view path) noexcept
{
  const std::size_t bytes = safetensors_weight_bytes(path);
  if (bytes == 0uz)
    return std::unexpected("Failed to load safetensors file or find supported tensors");
  constexpr std::size_t alignment_padding = kMaxTensors * (kSimdAlign - 1uz);
  if (bytes > std::numeric_limits<std::size_t>::max() - alignment_padding)
    return std::unexpected("Checkpoint weight storage size overflows this platform");
  try {
    std::shared_ptr<ModelWeights> weights{new ModelWeights{bytes}};
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
