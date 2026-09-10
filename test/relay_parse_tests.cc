#include "relay/tools/parse.h"

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <string_view>

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
  std::uint32_t value = 42u;
  float number = 1.25F;

  EXPECT_FALSE(parse_integer("42ms", value));
  EXPECT_EQ(value, 42u);
  EXPECT_FALSE(parse_integer("", value));
  EXPECT_EQ(value, 42u);
  EXPECT_FALSE(parse_number("1.25f", number));
  EXPECT_FLOAT_EQ(number, 1.25F);
  EXPECT_FALSE(parse_number("", number));
  EXPECT_FLOAT_EQ(number, 1.25F);
}

TEST(RelayParse, RejectsUnsignedSignsAndIntegerOverflowWithoutMutation)
{
  const std::uint32_t original = 7u;
  std::uint32_t value = original;

  EXPECT_FALSE(parse_integer("4294967296", value));
  EXPECT_EQ(value, original);
  EXPECT_FALSE(parse_integer("-1", value));
  EXPECT_EQ(value, original);
  EXPECT_FALSE(parse_integer("+1", value));
  EXPECT_EQ(value, original);
}

TEST(RelayParse, RejectsNonFiniteFloatingPointValues)
{
  float value = 3.5F;

  for (const std::string_view text : {"nan", "NaN", "inf", "INF", "-inf"}) {
    EXPECT_FALSE(parse_number(text, value)) << text;
    EXPECT_EQ(value, 3.5F) << text;
  }
}

TEST(RelayParse, RejectsFloatingOverflowWithoutChangingTheDestination)
{
  float value = 3.5F;

  EXPECT_FALSE(parse_number("1e39", value));
  EXPECT_EQ(value, 3.5F);
  EXPECT_FALSE(parse_number("-1e39", value));
  EXPECT_EQ(value, 3.5F);
}

TEST(RelayParse, EnforcesSignedIntegerRanges)
{
  std::int8_t value = 7;

  EXPECT_TRUE(parse_integer("-128", value));
  EXPECT_EQ(value, -128);
  EXPECT_TRUE(parse_integer("127", value));
  EXPECT_EQ(value, 127);
  for (const std::string_view text : {"-129", "128"}) {
    EXPECT_FALSE(parse_integer(text, value)) << text;
    EXPECT_EQ(value, 127) << text;
  }
}

TEST(RelayParse, EnforcesUnsignedIntegerRanges)
{
  std::uint8_t value = 7u;

  EXPECT_TRUE(parse_integer("255", value));
  EXPECT_EQ(value, 255u);
  for (const std::string_view text : {"256", "-1"}) {
    EXPECT_FALSE(parse_integer(text, value)) << text;
    EXPECT_EQ(value, 255u) << text;
  }
}

TEST(RelayParse, AcceptsSignedValuesAtFloatingPointLimits)
{
  float value{};

  EXPECT_TRUE(parse_number("-3.5", value));
  EXPECT_FLOAT_EQ(value, -3.5F);
  EXPECT_TRUE(parse_number("1.0e+0", value));
  EXPECT_FLOAT_EQ(value, 1.0F);
  EXPECT_TRUE(parse_number("3.4028234e38", value));
  EXPECT_TRUE(std::isfinite(value));
}

} // namespace
