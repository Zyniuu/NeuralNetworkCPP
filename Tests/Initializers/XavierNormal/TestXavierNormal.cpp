/**
 * C++ neural network library
 *
 * TestXavierNormal.cpp
 */

#include <gtest/gtest.h>
#include <numeric>
#include <NeuralNetworkCPP/Initializers/XavierNormal/XavierNormal.hpp>
#include "../../TestUtils/TestUtils.hpp"

// Test whether XavierNormal generates values within an expected range
TEST(XavierNormalTests, ValueRange)
{
    int inputs = 100;
    int outputs = 50;
    nn::XavierNormal initializer(inputs, outputs);

    double expectedStdDev = std::sqrt(2.0 / (inputs + outputs));
    std::vector<double> values(1000);

    for (double &v : values)
    {
        v = initializer.getRandomNum();
    }

    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double stdDev = calculateStdDev(values, mean);

    EXPECT_NEAR(mean, 0.0, 0.1);              // Mean should be around 0
    EXPECT_NEAR(stdDev, expectedStdDev, 0.1); // Standard deviation should match expected
}

// Test that different instances produce different results
TEST(XavierNormalTests, Randomness)
{
    int inputs = 100;
    int outputs = 50;
    nn::XavierNormal init1(inputs, outputs);
    nn::XavierNormal init2(inputs, outputs);

    double val1 = init1.getRandomNum();
    double val2 = init2.getRandomNum();

    EXPECT_NE(val1, val2); // Not guaranteed to always pass, but usually should
}

// Test that small inputs and outputs do not cause issues
TEST(XavierNormalTests, SmallInputsOutputs)
{
    int inputs = 1;
    int outputs = 1;
    nn::XavierNormal initializer(inputs, outputs);

    EXPECT_NO_THROW(initializer.getRandomNum());
}