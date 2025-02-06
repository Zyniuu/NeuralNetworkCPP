/**
 * C++ neural network library
 *
 * TestActivations.cpp
 */

#include <gtest/gtest.h>
#include "../../NeuralNetworkCPP/Activations/ReLU/ReLU.hpp"

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