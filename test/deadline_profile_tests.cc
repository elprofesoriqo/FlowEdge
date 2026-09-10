#include "bench/deadline_profile.h"

#include <array>
#include <gtest/gtest.h>

TEST(DeadlineProfile, UsesNearestRankPercentiles)
{
  constexpr std::array<double, 10> samples{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
  const fe::benchmark::Statistics stats = fe::benchmark::summarize(samples);
  EXPECT_DOUBLE_EQ(stats.mean, 5.5);
  EXPECT_DOUBLE_EQ(stats.p50, 5.0);
  EXPECT_DOUBLE_EQ(stats.p95, 10.0);
  EXPECT_DOUBLE_EQ(stats.p99, 10.0);
  EXPECT_DOUBLE_EQ(stats.p999, 10.0);
  EXPECT_DOUBLE_EQ(stats.max, 10.0);
}

TEST(DeadlineProfile, HandlesEmptySamples)
{
  const fe::benchmark::Statistics stats = fe::benchmark::summarize(std::span<const double>{});
  EXPECT_DOUBLE_EQ(stats.mean, 0.0);
  EXPECT_DOUBLE_EQ(stats.max, 0.0);
}

TEST(DeadlineProfile, ComputesRegressionAndBudget)
{
  constexpr std::array<double, 4> samples{10.0, 20.0, 30.0, 40.0};
  const fe::benchmark::Statistics stats = fe::benchmark::summarize(samples);
  EXPECT_TRUE(fe::benchmark::within_budget(stats, 40.0, stats.max));
  EXPECT_FALSE(fe::benchmark::within_budget(stats, 39.0, stats.max));
  EXPECT_DOUBLE_EQ(fe::benchmark::regression_percent(100.0, 105.0), 5.0);
  EXPECT_DOUBLE_EQ(fe::benchmark::regression_percent(100.0, 95.0), -5.0);
}
