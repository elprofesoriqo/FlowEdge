#pragma once

#include <cstdint>
#include <string_view>

namespace fe::relay {

enum class ClientResult : std::uint8_t
{
  kSuccess,
  kEmpty,
  kFull,
  kInvalidRequest,
  kIncompatible,
  kCorrupt,
  kTransportError,
};

[[nodiscard]] std::string_view to_string(ClientResult result) noexcept;

} // namespace fe::relay
