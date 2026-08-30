#include "relay/worker/worker_topology.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <cstdio>
#include <sched.h>
#endif

namespace fe::relay {
namespace {

struct CpuLocation
{
  std::uint32_t node{};
  std::uint32_t cpu{};
#ifdef _WIN32
  std::uint16_t group{};
#endif
};

struct Topology
{
  static constexpr std::size_t kMaxCpus = 256uz;
  std::array<CpuLocation, kMaxCpus> cpus{};
  std::size_t size{};
  std::size_t nodes{};
};

#ifdef __linux__
[[nodiscard]] bool cpu_list_contains(const char* text, std::uint32_t cpu) noexcept
{
  const char* cursor = text;
  while (*cursor != '\0') {
    char* end{};
    const unsigned long first = std::strtoul(cursor, &end, 10);
    if (end == cursor)
      break;
    unsigned long last = first;
    cursor = end;
    if (*cursor == '-') {
      last = std::strtoul(cursor + 1, &end, 10);
      cursor = end;
    }
    if (cpu >= first && cpu <= last)
      return true;
    while (*cursor == ',' || *cursor == ' ' || *cursor == '\n')
      ++cursor;
  }
  return false;
}

[[nodiscard]] std::uint32_t linux_numa_node(std::uint32_t cpu) noexcept
{
  std::array<char, 96uz> path{};
  std::array<char, 4096uz> list{};
  for (std::uint32_t node{}; node < 256u; ++node) {
    const int length =
        std::snprintf(path.data(), path.size(), "/sys/devices/system/node/node%u/cpulist", node);
    if (length <= 0 || static_cast<std::size_t>(length) >= path.size())
      continue;
    FILE* const file = std::fopen(path.data(), "r");
    if (file == nullptr)
      continue;
    const bool read = std::fgets(list.data(), static_cast<int>(list.size()), file) != nullptr;
    std::fclose(file);
    if (read && cpu_list_contains(list.data(), cpu))
      return node;
  }
  return 0u;
}
#endif

[[nodiscard]] Topology discover_topology() noexcept
{
  Topology topology{};
#ifdef _WIN32
  ULONG highest_node{};
  if (!GetNumaHighestNodeNumber(&highest_node))
    highest_node = 0u;
  for (ULONG node{}; node <= highest_node && topology.size < topology.cpus.size(); ++node) {
    GROUP_AFFINITY affinity{};
    if (!GetNumaNodeProcessorMaskEx(static_cast<USHORT>(node), &affinity))
      continue;
    for (unsigned bit{}; bit < sizeof(KAFFINITY) * 8u && topology.size < topology.cpus.size();
         ++bit) {
      if ((affinity.Mask & (static_cast<KAFFINITY>(1u) << bit)) == 0u)
        continue;
      topology.cpus[topology.size++] = CpuLocation{.node = static_cast<std::uint32_t>(node),
                                                   .cpu = bit,
                                                   .group = affinity.Group};
    }
  }
#elif defined(__linux__)
  cpu_set_t allowed{};
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
    for (unsigned cpu{}; cpu < CPU_SETSIZE && topology.size < topology.cpus.size(); ++cpu)
      if (CPU_ISSET(cpu, &allowed))
        topology.cpus[topology.size++] = CpuLocation{.node = linux_numa_node(cpu), .cpu = cpu};
  }
#endif
  if (topology.size == 0uz)
    return topology;
  std::ranges::sort(topology.cpus.begin(), topology.cpus.begin() + topology.size,
                    [](const CpuLocation& left, const CpuLocation& right) {
                      return left.node < right.node ||
                             (left.node == right.node && left.cpu < right.cpu);
                    });
  topology.nodes = 1uz;
  for (std::size_t index{1uz}; index < topology.size; ++index)
    topology.nodes += topology.cpus[index].node != topology.cpus[index - 1uz].node ? 1uz : 0uz;
  return topology;
}

[[nodiscard]] const CpuLocation* select_cpu(const Topology& topology, std::size_t worker_index,
                                            WorkerPlacement placement) noexcept
{
  if (topology.size == 0uz)
    return nullptr;
  if (placement == WorkerPlacement::kCompact)
    return &topology.cpus[worker_index % topology.size];

  std::array<std::uint32_t, 256uz> nodes{};
  std::size_t node_count{};
  for (std::size_t index{}; index < topology.size; ++index)
    if (index == 0uz || topology.cpus[index].node != topology.cpus[index - 1uz].node)
      nodes[node_count++] = topology.cpus[index].node;
  const std::uint32_t target_node = nodes[worker_index % node_count];
  const std::size_t lane = worker_index / node_count;
  std::size_t count{};
  const CpuLocation* fallback{};
  for (std::size_t index{}; index < topology.size; ++index) {
    if (topology.cpus[index].node != target_node)
      continue;
    if (fallback == nullptr)
      fallback = &topology.cpus[index];
    if (count++ == lane)
      return &topology.cpus[index];
  }
  return fallback;
}

} // namespace

std::expected<WorkerPlacement, std::string_view> parse_worker_placement(
    std::string_view value) noexcept
{
  if (value == "none")
    return WorkerPlacement::kNone;
  if (value == "compact")
    return WorkerPlacement::kCompact;
  if (value == "spread")
    return WorkerPlacement::kSpread;
  return std::unexpected("worker placement must be none, compact, or spread");
}

std::string_view to_string(WorkerPlacement placement) noexcept
{
  switch (placement) {
  case WorkerPlacement::kNone:
    return "none";
  case WorkerPlacement::kCompact:
    return "compact";
  case WorkerPlacement::kSpread:
    return "spread";
  }
  return "unknown";
}

std::size_t available_numa_nodes() noexcept
{
  return discover_topology().nodes;
}

WorkerBinding bind_current_worker(std::size_t worker_index, WorkerPlacement placement) noexcept
{
  if (placement == WorkerPlacement::kNone)
    return {};
  const Topology topology = discover_topology();
  const CpuLocation* const selected = select_cpu(topology, worker_index, placement);
  if (selected == nullptr)
    return {};

  bool bound{};
#ifdef _WIN32
  GROUP_AFFINITY affinity{};
  affinity.Group = selected->group;
  affinity.Mask = static_cast<KAFFINITY>(1u) << selected->cpu;
  bound = SetThreadGroupAffinity(GetCurrentThread(), &affinity, nullptr) != 0;
#elif defined(__linux__)
  cpu_set_t affinity{};
  CPU_ZERO(&affinity);
  CPU_SET(selected->cpu, &affinity);
  bound = sched_setaffinity(0, sizeof(affinity), &affinity) == 0;
#endif
  return WorkerBinding{.numa_node = selected->node, .logical_cpu = selected->cpu, .bound = bound};
}

} // namespace fe::relay
