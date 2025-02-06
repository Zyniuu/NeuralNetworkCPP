/**
 * C++ neural network library
 *
 * TestActivations.cpp
 */

#include <gtest/gtest.h>
#include "../../NeuralNetworkCPP/Activations/ReLU/ReLU.hpp"
#include "../../NeuralNetworkCPP/Activations/Sigmoid/Sigmoid.hpp"

TEST(ActivationsTests, ReLU)
{
    nn::ReLU relu;

    nn::Matrix input(2, 2, {1.0, -2.0, 3.0, 0.0});
    nn::Matrix output = relu.forward(input);

    // Verify forward pass
    EXPECT_DOUBLE_EQ(output(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(output(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(output(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(output(1, 1), 0.0);

    nn::Matrix gradOutput(2, 2, {0.1, 0.2, 0.0, -0.4});
    nn::Matrix gradInput = relu.backward(gradOutput);

    // Verify backward pass
    EXPECT_DOUBLE_EQ(gradInput(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(gradInput(0, 1), 1.0);
    EXPECT_DOUBLE_EQ(gradInput(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(gradInput(1, 1), 0.0);
}

TEST(ActivationsTests, Sigmoid)
{
    nn::Sigmoid sigmoid;

    nn::Matrix input(2, 2, {0.0, 1.0, -1.0, 2.0});
    nn::Matrix output = sigmoid.forward(input);

    // Verify forward pass
    EXPECT_NEAR(output(0, 0), 0.5, 1e-1);
    EXPECT_NEAR(output(0, 1), 0.731059, 1e-6);
    EXPECT_NEAR(output(1, 0), 0.268941, 1e-6);
    EXPECT_NEAR(output(1, 1), 0.880797, 1e-6);

    nn::Matrix gradOutput(2, 2, {0.1, 0.2, 0.3, 0.4});
    nn::Matrix gradInput = sigmoid.backward(gradOutput);

    // Verify backward pass
    EXPECT_NEAR(gradInput(0, 0), 0.025, 1e-3);
    EXPECT_NEAR(gradInput(0, 1), 0.03932, 1e-5);
    EXPECT_NEAR(gradInput(1, 0), 0.05898, 1e-5);
    EXPECT_NEAR(gradInput(1, 1), 0.04199, 1e-5);
}