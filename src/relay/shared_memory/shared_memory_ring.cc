#include "relay/shared_memory/shared_memory_ring.h"

#include "protocol/model_identity.h"

#include <array>
#include <atomic>
#include <bit>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace fe::relay {
namespace {

constexpr std::uint32_t kRingMagic = 0x31475246u; // "FRG1"
constexpr std::uint32_t kRingVersion = 1u;
constexpr std::size_t kAlignment = 64uz;

struct alignas(kAlignment) RingHeader
{
  std::uint32_t magic{kRingMagic};
  std::uint32_t version{kRingVersion};
  std::uint32_t capacity{};
  std::uint32_t slot_bytes{};
  std::uint64_t mapping_bytes{};
  std::array<std::byte, kAlignment - 24uz> first_padding{};
  alignas(kAlignment) std::atomic<std::uint64_t> write_sequence{0u};
  std::array<std::byte, kAlignment - sizeof(std::atomic<std::uint64_t>)> write_padding{};
  alignas(kAlignment) std::atomic<std::uint64_t> read_sequence{0u};
  std::array<std::byte, kAlignment - sizeof(std::atomic<std::uint64_t>)> read_padding{};
};
static_assert(sizeof(RingHeader) == 3uz * kAlignment);
static_assert(std::atomic<std::uint64_t>::is_always_lock_free);

struct SlotHeader
{
  std::uint32_t bytes{};
  std::uint32_t reserved{};
  std::uint64_t checksum{};
};

[[nodiscard]] constexpr std::size_t align_up(std::size_t value) noexcept
{
  return (value + kAlignment - 1uz) & ~(kAlignment - 1uz);
}

[[nodiscard]] constexpr std::size_t slot_stride(std::uint32_t slot_bytes) noexcept
{
  return align_up(sizeof(SlotHeader) + slot_bytes);
}

[[nodiscard]] constexpr std::size_t required_bytes(RingConfig config) noexcept
{
  if (config.capacity < 2u || !std::has_single_bit(config.capacity) || config.slot_bytes == 0u)
    return 0uz;
  const std::size_t stride = slot_stride(config.slot_bytes);
  if (stride > (std::numeric_limits<std::size_t>::max() - sizeof(RingHeader)) / config.capacity)
    return 0uz;
  return sizeof(RingHeader) + (stride * config.capacity);
}

[[nodiscard]] RingHeader* header(void* mapping) noexcept
{
  return static_cast<RingHeader*>(mapping);
}

[[nodiscard]] const RingHeader* header(const void* mapping) noexcept
{
  return static_cast<const RingHeader*>(mapping);
}

[[nodiscard]] SlotHeader* slot(void* mapping, std::uint64_t sequence) noexcept
{
  RingHeader* const state = header(mapping);
  const auto index = static_cast<std::size_t>(sequence & (state->capacity - 1u));
  auto* const base = static_cast<std::byte*>(mapping) + sizeof(RingHeader);
  return reinterpret_cast<SlotHeader*>(base + (index * slot_stride(state->slot_bytes)));
}

[[nodiscard]] std::uint64_t checksum(std::span<const std::byte> payload) noexcept
{
  const ModelDigest digest = fingerprint_bytes(payload);
  std::uint64_t value{0u};
  for (std::size_t i{0uz}; i < sizeof(value); ++i)
    value |= static_cast<std::uint64_t>(digest[i]) << (8uz * i);
  return value;
}

#ifndef _WIN32
[[nodiscard]] std::string posix_name(std::string name)
{
  if (name.empty() || name.front() != '/')
    name.insert(name.begin(), '/');
  for (std::size_t i{1uz}; i < name.size(); ++i)
    if (name[i] == '/')
      name[i] = '_';
  return name;
}
#endif

[[nodiscard]] std::expected<void, std::string> validate_mapping(void* mapping,
                                                                std::size_t bytes) noexcept
{
  if (mapping == nullptr || bytes < sizeof(RingHeader))
    return std::unexpected("Shared-memory ring is smaller than its header");
  const RingHeader* const state = header(mapping);
  const RingConfig config{.capacity = state->capacity, .slot_bytes = state->slot_bytes};
  const std::size_t expected = required_bytes(config);
  if (state->magic != kRingMagic || state->version != kRingVersion || expected == 0uz ||
      state->mapping_bytes != expected || bytes < expected)
    return std::unexpected("Shared-memory ring header is incompatible or corrupt");
  if (!state->write_sequence.is_lock_free() || !state->read_sequence.is_lock_free())
    return std::unexpected("Interprocess 64-bit atomics are not lock-free on this platform");
  return {};
}

} // namespace

std::expected<SharedMemoryRing, std::string> SharedMemoryRing::create(
    std::string name, RingConfig config, bool unlink_on_destroy) noexcept
{
  const std::size_t bytes = required_bytes(config);
  if (name.empty() || bytes == 0uz)
    return std::unexpected("Ring name, power-of-two capacity, or slot size is invalid");

  SharedMemoryRing ring;
#ifdef _WIN32
  const std::uint64_t wide_bytes = bytes;
  HANDLE handle = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                                     static_cast<DWORD>(wide_bytes >> 32u),
                                     static_cast<DWORD>(wide_bytes & 0xffffffffu), name.c_str());
  if (handle == nullptr)
    return std::unexpected("CreateFileMapping failed for shared-memory ring");
  if (GetLastError() == ERROR_ALREADY_EXISTS) {
    CloseHandle(handle);
    return std::unexpected("Shared-memory ring already exists");
  }
  void* mapping = MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, bytes);
  if (mapping == nullptr) {
    CloseHandle(handle);
    return std::unexpected("MapViewOfFile failed for shared-memory ring");
  }
  ring.handle_ = handle;
#else
  name = posix_name(std::move(name));
  const int descriptor = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
  if (descriptor < 0)
    return std::unexpected("shm_open create failed for shared-memory ring");
  if (ftruncate(descriptor, static_cast<off_t>(bytes)) != 0) {
    close(descriptor);
    shm_unlink(name.c_str());
    return std::unexpected("ftruncate failed for shared-memory ring");
  }
  void* mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, descriptor, 0);
  if (mapping == MAP_FAILED) {
    close(descriptor);
    shm_unlink(name.c_str());
    return std::unexpected("mmap failed for shared-memory ring");
  }
  ring.descriptor_ = descriptor;
#endif
  std::memset(mapping, 0, bytes);
  auto* const state = std::construct_at(static_cast<RingHeader*>(mapping));
  state->capacity = config.capacity;
  state->slot_bytes = config.slot_bytes;
  state->mapping_bytes = bytes;
  state->write_sequence.store(0u, std::memory_order_relaxed);
  state->read_sequence.store(0u, std::memory_order_relaxed);
  ring.mapping_ = mapping;
  ring.mapping_bytes_ = bytes;
  ring.name_ = std::move(name);
  ring.unlink_on_destroy_ = unlink_on_destroy;
  return ring;
}

std::expected<SharedMemoryRing, std::string> SharedMemoryRing::open(std::string name) noexcept
{
  if (name.empty())
    return std::unexpected("Shared-memory ring name is empty");
  SharedMemoryRing ring;
  std::size_t bytes{0uz};
#ifdef _WIN32
  HANDLE handle = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, name.c_str());
  if (handle == nullptr)
    return std::unexpected("OpenFileMapping failed for shared-memory ring");
  void* mapping = MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, 0);
  if (mapping == nullptr) {
    CloseHandle(handle);
    return std::unexpected("MapViewOfFile failed for shared-memory ring");
  }
  MEMORY_BASIC_INFORMATION mapping_info{};
  if (VirtualQuery(mapping, &mapping_info, sizeof(mapping_info)) == 0u) {
    UnmapViewOfFile(mapping);
    CloseHandle(handle);
    return std::unexpected("VirtualQuery failed for shared-memory ring");
  }
  bytes = mapping_info.RegionSize;
  ring.handle_ = handle;
#else
  name = posix_name(std::move(name));
  const int descriptor = shm_open(name.c_str(), O_RDWR, 0600);
  if (descriptor < 0)
    return std::unexpected("shm_open failed for shared-memory ring");
  struct stat info = {};
  if (fstat(descriptor, &info) != 0 || info.st_size < 0) {
    close(descriptor);
    return std::unexpected("fstat failed for shared-memory ring");
  }
  bytes = static_cast<std::size_t>(info.st_size);
  void* mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, descriptor, 0);
  if (mapping == MAP_FAILED) {
    close(descriptor);
    return std::unexpected("mmap failed for shared-memory ring");
  }
  ring.descriptor_ = descriptor;
#endif
  const auto valid = validate_mapping(mapping, bytes);
  if (!valid) {
#ifdef _WIN32
    UnmapViewOfFile(mapping);
    CloseHandle(static_cast<HANDLE>(ring.handle_));
    ring.handle_ = nullptr;
#else
    munmap(mapping, bytes);
    close(ring.descriptor_);
    ring.descriptor_ = -1;
#endif
    return std::unexpected(valid.error());
  }
  ring.mapping_ = mapping;
  ring.mapping_bytes_ = bytes;
  ring.name_ = std::move(name);
  return ring;
}

SharedMemoryRing::~SharedMemoryRing()
{
  release();
}

SharedMemoryRing::SharedMemoryRing(SharedMemoryRing&& other) noexcept
{
  *this = std::move(other);
}

SharedMemoryRing& SharedMemoryRing::operator=(SharedMemoryRing&& other) noexcept
{
  if (this == &other)
    return *this;
  release();
  mapping_ = std::exchange(other.mapping_, nullptr);
  mapping_bytes_ = std::exchange(other.mapping_bytes_, 0uz);
  name_ = std::move(other.name_);
  unlink_on_destroy_ = std::exchange(other.unlink_on_destroy_, false);
#ifdef _WIN32
  handle_ = std::exchange(other.handle_, nullptr);
#else
  descriptor_ = std::exchange(other.descriptor_, -1);
#endif
  return *this;
}

RingConfig SharedMemoryRing::config() const noexcept
{
  return valid() ? RingConfig{.capacity = header(mapping_)->capacity,
                              .slot_bytes = header(mapping_)->slot_bytes}
                 : RingConfig{};
}

std::uint64_t SharedMemoryRing::size() const noexcept
{
  if (!valid())
    return 0u;
  const RingHeader* const state = header(mapping_);
  return state->write_sequence.load(std::memory_order_acquire) -
         state->read_sequence.load(std::memory_order_acquire);
}

RingResult SharedMemoryRing::try_push(std::span<const std::byte> payload) noexcept
{
  if (!valid() || payload.size() > header(mapping_)->slot_bytes)
    return RingResult::kTooLarge;
  RingHeader* const state = header(mapping_);
  const std::uint64_t write = state->write_sequence.load(std::memory_order_relaxed);
  const std::uint64_t read = state->read_sequence.load(std::memory_order_acquire);
  if (write - read >= state->capacity)
    return RingResult::kFull;
  SlotHeader* const destination = slot(mapping_, write);
  destination->bytes = static_cast<std::uint32_t>(payload.size());
  destination->reserved = 0u;
  destination->checksum = checksum(payload);
  std::memcpy(destination + 1, payload.data(), payload.size());
  state->write_sequence.store(write + 1u, std::memory_order_release);
  return RingResult::kSuccess;
}

RingResult SharedMemoryRing::try_pop(std::span<std::byte> destination,
                                     std::size_t& bytes_read) noexcept
{
  bytes_read = 0uz;
  if (!valid())
    return RingResult::kCorrupt;
  RingHeader* const state = header(mapping_);
  const std::uint64_t read = state->read_sequence.load(std::memory_order_relaxed);
  const std::uint64_t write = state->write_sequence.load(std::memory_order_acquire);
  if (read == write)
    return RingResult::kEmpty;
  SlotHeader* const source = slot(mapping_, read);
  if (source->bytes > state->slot_bytes || source->reserved != 0u) {
    state->read_sequence.store(read + 1u, std::memory_order_release);
    return RingResult::kCorrupt;
  }
  if (destination.size() < source->bytes)
    return RingResult::kDestinationTooSmall;
  const std::span<const std::byte> payload{reinterpret_cast<const std::byte*>(source + 1),
                                           source->bytes};
  if (source->checksum != checksum(payload)) {
    state->read_sequence.store(read + 1u, std::memory_order_release);
    return RingResult::kCorrupt;
  }
  std::memcpy(destination.data(), payload.data(), payload.size());
  bytes_read = payload.size();
  state->read_sequence.store(read + 1u, std::memory_order_release);
  return RingResult::kSuccess;
}

void SharedMemoryRing::release() noexcept
{
  if (mapping_ != nullptr) {
#ifdef _WIN32
    UnmapViewOfFile(mapping_);
#else
    munmap(mapping_, mapping_bytes_);
#endif
    mapping_ = nullptr;
  }
#ifdef _WIN32
  if (handle_ != nullptr) {
    CloseHandle(static_cast<HANDLE>(handle_));
    handle_ = nullptr;
  }
#else
  if (descriptor_ >= 0) {
    close(descriptor_);
    descriptor_ = -1;
  }
  if (unlink_on_destroy_ && !name_.empty())
    shm_unlink(name_.c_str());
#endif
  mapping_bytes_ = 0uz;
  unlink_on_destroy_ = false;
}

} // namespace fe::relay
