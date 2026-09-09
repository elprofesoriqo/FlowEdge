#include "protocol/byte_codec.h"

#include <array>
#include <cstdint>
#include <gtest/gtest.h>

namespace {

TEST(ByteCodec, WritesCanonicalLittleEndianBytes)
{
  std::array<std::byte, 8> bytes{};

  fe::protocol::write_le(bytes, 0uz, std::uint16_t{0x1234u});
  fe::protocol::write_le(bytes, 2uz, std::uint32_t{0x89abcdefu});
  fe::protocol::write_le(bytes, 6uz, std::uint16_t{0xfedcu});

  EXPECT_EQ(bytes[0], std::byte{0x34});
  EXPECT_EQ(bytes[1], std::byte{0x12});
  EXPECT_EQ(bytes[2], std::byte{0xef});
  EXPECT_EQ(bytes[3], std::byte{0xcd});
  EXPECT_EQ(bytes[4], std::byte{0xab});
  EXPECT_EQ(bytes[5], std::byte{0x89});
  EXPECT_EQ(bytes[6], std::byte{0xdc});
  EXPECT_EQ(bytes[7], std::byte{0xfe});
}

TEST(ByteCodec, RoundTripsSignedAndUnsignedWidths)
{
  std::array<std::byte, 16> bytes{};
  fe::protocol::write_le(bytes, 0uz, std::int8_t{-7});
  fe::protocol::write_le(bytes, 1uz, std::int16_t{-1234});
  fe::protocol::write_le(bytes, 3uz, std::uint32_t{0xdeadbeefu});
  fe::protocol::write_le(bytes, 7uz, std::int64_t{-9876543210ll});

  EXPECT_EQ(fe::protocol::read_le<std::int8_t>(bytes, 0uz), -7);
  EXPECT_EQ(fe::protocol::read_le<std::int16_t>(bytes, 1uz), -1234);
  EXPECT_EQ(fe::protocol::read_le<std::uint32_t>(bytes, 3uz), 0xdeadbeefu);
  EXPECT_EQ(fe::protocol::read_le<std::int64_t>(bytes, 7uz), -9876543210ll);
}

} // namespace
