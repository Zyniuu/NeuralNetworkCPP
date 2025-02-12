/**
 * C++ neural network library
 *
 * TestActivations.cpp
 */

#include <gtest/gtest.h>
#include <NeuralNetworkCPP/Activations/Activations.hpp>

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
    EXPECT_NEAR(gradInput(0, 0), 0.24937, 1e-5);
    EXPECT_NEAR(gradInput(0, 1), 0.24751, 1e-5);
    EXPECT_NEAR(gradInput(1, 0), 0.24445, 1e-5);
    EXPECT_NEAR(gradInput(1, 1), 0.24026, 1e-5);
}

TEST(ActivationsTests, Softmax)
{
    nn::Softmax softmax;

    nn::Matrix input(2, 3, {1.0, 2.0, 3.0, 1.0, 2.0, 3.0});
    nn::Matrix output = softmax.forward(input);

    // Verify forward pass
    EXPECT_NEAR(output(0, 0), 0.04501, 1e-5);
    EXPECT_NEAR(output(0, 1), 0.12236, 1e-5);
    EXPECT_NEAR(output(0, 2), 0.33262, 1e-5);
    EXPECT_NEAR(output(1, 0), 0.04501, 1e-5);
    EXPECT_NEAR(output(1, 1), 0.12236, 1e-5);
    EXPECT_NEAR(output(1, 2), 0.33262, 1e-5);

    nn::Matrix gradOutput(2, 3, {0.1, 0.2, 0.3, 0.1, 0.2, 0.3});
    nn::Matrix gradInput = softmax.backward(gradOutput);

    // Verify backward pass
    EXPECT_NEAR(gradInput(0, 0), 0.00429, 1e-5);
    EXPECT_NEAR(gradInput(0, 1), 0.02147, 1e-5);
    EXPECT_NEAR(gradInput(0, 2), 0.06659, 1e-5);
    EXPECT_NEAR(gradInput(1, 0), 0.00429, 1e-5);
    EXPECT_NEAR(gradInput(1, 1), 0.02147, 1e-5);
    EXPECT_NEAR(gradInput(1, 2), 0.06659, 1e-5);
}