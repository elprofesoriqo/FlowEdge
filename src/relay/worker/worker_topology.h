#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string_view>

namespace fe::relay {

enum class WorkerPlacement : std::uint8_t
{
  kNone,
  kCompact,
  kSpread,
};

struct WorkerBinding
{
  std::uint32_t numa_node{};
  std::uint32_t logical_cpu{};
  bool bound{};
};

[[nodiscard]] std::expected<WorkerPlacement, std::string_view> parse_worker_placement(
    std::string_view value) noexcept;
[[nodiscard]] std::string_view to_string(WorkerPlacement placement) noexcept;
[[nodiscard]] std::size_t available_numa_nodes() noexcept;
[[nodiscard]] WorkerBinding bind_current_worker(std::size_t worker_index,
                                                WorkerPlacement placement) noexcept;

} // namespace fe::relay
