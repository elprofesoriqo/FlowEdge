#include "safetensors.h"

#include <algorithm>
#include <array>
#include <cstring>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace fe {
namespace {

struct MappedFile
{
  const std::byte* data{};
  std::size_t size{0uz};

#ifdef _WIN32
  HANDLE file{INVALID_HANDLE_VALUE}, mapping{};

  [[nodiscard]] bool open(const char* p) noexcept
  {
    file = CreateFileA(p, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                       FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE)
      return false;
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(file, &sz)) {
      close();
      return false;
    }
    size = static_cast<std::size_t>(sz.QuadPart);
    mapping = CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping) {
      close();
      return false;
    }
    data = static_cast<const std::byte*>(MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0));
    if (!data) {
      close();
      return false;
    }
    return true;
  }
  void close() noexcept
  {
    if (data)
      UnmapViewOfFile(data);
    if (mapping)
      CloseHandle(mapping);
    if (file != INVALID_HANDLE_VALUE)
      CloseHandle(file);
  }
#else
  int fd{-1};

  [[nodiscard]] bool open(const char* p) noexcept
  {
    fd = ::open(p, O_RDONLY);
    if (fd < 0)
      return false;
    struct stat st{};
    if (fstat(fd, &st) < 0) {
      close();
      return false;
    }
    size = static_cast<std::size_t>(st.st_size);
    void* m = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) {
      close();
      return false;
    }
    data = static_cast<const std::byte*>(m);
    return true;
  }
  void close() noexcept
  {
    if (data)
      munmap(const_cast<std::byte*>(data), size);
    if (fd >= 0)
      ::close(fd);
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
  auto p = json.begin(), end = json.end();
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
    if (obj.find("\"F32\"") == std::string_view::npos)
      continue;

    std::array<std::uint64_t, 4> shape{};
    auto ndim = parse_u64s(after_colon(obj, "\"shape\""), std::span{shape});
    std::array<std::uint64_t, 2> offs{};
    if (parse_u64s(after_colon(obj, "\"data_offsets\""), std::span{offs}) < 2)
      continue;

    if (!cb(name, offs[0], offs[1] - offs[0], shape, static_cast<std::uint8_t>(ndim)))
      return false;
  }
  return true;
}

} // namespace

bool SafetensorsLoader::load(std::string_view path, Arena& arena, std::span<TensorView> out,
                             std::size_t& tensors_loaded) const noexcept
{
  tensors_loaded = 0uz;

  std::array<char, 512> buf{};
  if (path.size() >= buf.size())
    return false;
  std::memcpy(buf.data(), path.data(), path.size());

  MappedFile mf{};
  if (!mf.open(buf.data()))
    return false;

  if (mf.size < 8uz) {
    mf.close();
    return false;
  }

  std::uint64_t header_len{};
  std::memcpy(&header_len, mf.data, 8uz);
  if (header_len > mf.size - 8uz) { // overflow-safe: mf.size >= 8 here
    mf.close();
    return false;
  }

  const std::string_view json{reinterpret_cast<const char*>(mf.data + 8uz),
                              static_cast<std::size_t>(header_len)};
  const std::byte* weights_base = mf.data + 8uz + header_len;
  const std::uint64_t data_size = mf.size - 8uz - header_len;

  bool ok = foreach_tensor(json,
                           [&](std::string_view name, std::uint64_t byte_off,
                               std::uint64_t byte_len, const std::array<std::uint64_t, 4>& shape,
                               std::uint8_t ndim) noexcept -> bool {
                             if (tensors_loaded >= out.size()) [[unlikely]]
                               return false;
                             if (byte_len > data_size || byte_off > data_size - byte_len)
                                 [[unlikely]] // in-bounds
                               return false;

                             auto* dst =
                                 arena.alloc_array<float>(byte_len / sizeof(float), kSimdAlign);
                             if (!dst) [[unlikely]]
                               return false;
                             std::memcpy(dst, weights_base + byte_off, byte_len);

                             TensorView& tv = out[tensors_loaded++];
                             tv.data = dst;
                             tv.ndim = ndim;
                             for (std::size_t i{0uz}; i < 4uz; ++i)
                               tv.shape[i] = (i < ndim) ? static_cast<std::size_t>(shape[i]) : 0uz;
                             const std::size_t nlen = std::min(name.size(), tv.name.size() - 1uz);
                             std::memcpy(tv.name.data(), name.data(), nlen);
                             tv.name[nlen] = '\0';
                             return true;
                           });

  mf.close();
  return ok;
}

} // namespace fe
