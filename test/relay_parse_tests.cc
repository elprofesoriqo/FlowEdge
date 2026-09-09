#include "relay/tools/parse.h"

#include <cstdint>
#include <gtest/gtest.h>

namespace {

using fe::relay::cli::parse_integer;
using fe::relay::cli::parse_number;

TEST(RelayParse, AcceptsCompleteIntegerAndFloatingPointValues)
{
  std::uint64_t integer{};
  float number{};

  EXPECT_TRUE(parse_integer("1842", integer));
  EXPECT_EQ(integer, 1842u);
  EXPECT_TRUE(parse_number("1.25", number));
  EXPECT_FLOAT_EQ(number, 1.25f);
}

TEST(RelayParse, RejectsTrailingTextAndEmptyInput)
{
  std::uint32_t value{};

  EXPECT_FALSE(parse_integer("42ms", value));
  EXPECT_FALSE(parse_integer("", value));
}

TEST(RelayParse, RejectsOverflowWithoutChangingTheDestination)
{
  const std::uint32_t original = 7u;
  std::uint32_t value = original;

  EXPECT_FALSE(parse_integer("4294967296", value));
  EXPECT_EQ(value, original);
  EXPECT_FALSE(parse_integer("-1", value));
  EXPECT_EQ(value, original);
}

TEST(RelayParse, RejectsNonFiniteFloatingPointValues)
{
  float value = 3.5F;

  EXPECT_FALSE(parse_number("nan", value));
  EXPECT_EQ(value, 3.5F);
  EXPECT_FALSE(parse_number("inf", value));
  EXPECT_EQ(value, 3.5F);
  EXPECT_FALSE(parse_number("-inf", value));
  EXPECT_EQ(value, 3.5F);
}

} // namespace
