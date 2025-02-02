/**
 * C++ neural network library
 *
 * TestHeUniform.cpp
 */

#include <gtest/gtest.h>
#include <algorithm>
#include "../../../NeuralNetworkCPP/Initializers/HeUniform/HeUniform.hpp"
#include "../../TestUtils/TestUtils.hpp"

// Test whether HeUniform generates values with correct mean and variance
TEST(HeUniformTests, DistributionCheck)
{
    int inputs = 100;
    int outputs = 50;
    nn::HeUniform initializer(inputs, outputs);

    double expectedLimit = std::sqrt(6.0 / inputs);
    std::vector<double> values(1000);

    for (double &v : values)
    {
        v = initializer.getRandomNum();
    }

    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double stdDev = calculateStdDev(values, mean);

    double expectedStdDev = std::sqrt(2.0 / (3.0 * inputs)); // Unifrom standard deviation formula

    EXPECT_NEAR(mean, 0.0, 0.1);              // Mean should be around 0
    EXPECT_NEAR(stdDev, expectedStdDev, 0.1); // Standard deviation should match expected
}

// Test whether the generated values fall within the expected range
TEST(HeUniformTests, ValueRange)
{
    int inputs = 100;
    int outputs = 50;
    nn::HeUniform initializer(inputs, outputs);

    double limit = std::sqrt(6.0 / inputs); // Correct limit for He Uniform
    std::vector<double> values(1000);
    for (double &v : values)
    {
        v = initializer.getRandomNum();
    }

    // Check if at least 99% of values are in range
    int countWithinRange = std::count_if(values.begin(), values.end(), [&](double v)
        { return v >= -limit && v <= limit; }
    );

    double percentageWithinRange = static_cast<double>(countWithinRange) / values.size();

    EXPECT_GE(percentageWithinRange, 0.99);
}

// Test that different instances produce different results
TEST(HeUniformTests, Randomness)
{
    int inputs = 100;
    int outputs = 50;
    nn::HeUniform init1(inputs, outputs);
    nn::HeUniform init2(inputs, outputs);

    double val1 = init1.getRandomNum();
    double val2 = init2.getRandomNum();

    EXPECT_NE(val1, val2); // Not guaranteed to always pass, but usually should
}

// Test for small fan-in values
TEST(HeUniformTests, SmallInputs)
{
    int inputs = 1; // Very small input count
    int outputs = 1;
    nn::HeUniform initializer(inputs, outputs);

    EXPECT_NO_THROW(initializer.getRandomNum());
}