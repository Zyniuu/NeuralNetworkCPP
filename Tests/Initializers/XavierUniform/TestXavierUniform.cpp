/**
 * C++ neural network library
 *
 * TestXavierUniform.cpp
 */

#include <gtest/gtest.h>
#include <numeric>
#include <NeuralNetworkCPP/Initializers/XavierUniform/XavierUniform.hpp>
#include "../../TestUtils/TestUtils.hpp"

// Test whether XavierUniform generates values within an expected range
TEST(XavierUniformTests, ValueRange)
{
    int inputs = 100;
    int outputs = 50;
    nn::XavierUniform initializer(inputs, outputs);

    double limit = std::sqrt(6.0 / (inputs + outputs));
    std::vector<double> values(1000);

    for (double &v : values)
    {
        v = initializer.getRandomNum();
    }

    // Check if values are within [-limit, limit]
    for (double v : values)
    {
        EXPECT_GE(v, -limit);
        EXPECT_LE(v, limit);
    }
}

// Test that different instances produce different results
TEST(XavierUniformTests, Randomness)
{
    int inputs = 100;
    int outputs = 50;
    nn::XavierUniform init1(inputs, outputs);
    nn::XavierUniform init2(inputs, outputs);

    double val1 = init1.getRandomNum();
    double val2 = init2.getRandomNum();

    EXPECT_NE(val1, val2); // Not guaranteed to always pass, but usually should
}

// Test whether the generated numbers have a mean close to 0 and correct standard deviation
TEST(XavierUniformTests, DistributionCheck)
{
    int inputs = 100;
    int outputs = 50;
    nn::XavierUniform initializer(inputs, outputs);

    std::vector<double> values(1000);

    for (double &v : values)
    {
        v = initializer.getRandomNum();
    }

    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double stdDev = calculateStdDev(values, mean);

    double expectedStdDev = std::sqrt(2.0 / (inputs + outputs));

    EXPECT_NEAR(mean, 0.0, 0.1);              // Mean should be close to 0
    EXPECT_NEAR(stdDev, expectedStdDev, 0.1); // Standard deviation should match expected
}

// Test that small inputs and outputs do not cause issues
TEST(XavierUniformTests, SmallInputsOutputs)
{
    int inputs = 1;
    int outputs = 1;
    nn::XavierUniform initializer(inputs, outputs);

    EXPECT_NO_THROW(initializer.getRandomNum());
}