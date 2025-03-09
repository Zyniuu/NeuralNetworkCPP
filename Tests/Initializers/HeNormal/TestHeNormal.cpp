/**
 * C++ neural network library
 *
 * TestHeNormal.cpp
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <NeuralNetworkCPP/Initializers/HeNormal/HeNormal.hpp>
#include "../../TestUtils/TestUtils.hpp"

// Test whether HeNormal generates values with correct distribution
TEST(HeNormalTests, DistributionCheck)
{
    int inputs = 100;
    int outputs = 50; // Doesn't affect HeNormal
    nn::HeNormal initializer(inputs, outputs);

    double expectedStdDev = std::sqrt(2.0 / inputs);
    std::vector<double> values(1000);

    for (double &v : values)
    {
        v = initializer.getRandomNum();
    }

    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double stdDev = calculateStdDev(values, mean);

    EXPECT_NEAR(mean, 0.0, 0.1);              // Mean should be close to 0
    EXPECT_NEAR(stdDev, expectedStdDev, 0.1); // Standard deviation should match expected
}

TEST(HeNormalTests, ValueRange)
{
    int inputs = 100;
    int outputs = 50;
    nn::HeNormal initializer(inputs, outputs);

    double sigma = std::sqrt(2.0 / inputs);
    double range = 3 * sigma; // 99.7% of values should be in this range

    std::vector<double> values(1000);
    for (double &v : values)
    {
        v = initializer.getRandomNum();
    }

    // Count the values in range
    int countWithinRange = std::count_if(values.begin(), values.end(), [&](double v)
        { return v >= -range && v <= range; }
    );

    double percentageWithinRange = static_cast<double>(countWithinRange) / values.size();

    // Check if at least 99% of values are in range
    EXPECT_GE(percentageWithinRange, 0.99);
}

// Test that different instances produce different results
TEST(HeNormalTests, Randomness)
{
    int inputs = 100;
    int outputs = 50; // Doesn't affect HeNormal
    nn::HeNormal init1(inputs, outputs);
    nn::HeNormal init2(inputs, outputs);

    double val1 = init1.getRandomNum();
    double val2 = init2.getRandomNum();

    EXPECT_NE(val1, val2); // Not guaranteed to always pass, but usually should
}

// Test for small inputs values
TEST(HeNormalTests, SmallInputs)
{
    int inputs = 1; // Small inputs count
    int outputs = 1;
    nn::HeNormal initializer(inputs, outputs);

    EXPECT_NO_THROW(initializer.getRandomNum());
}