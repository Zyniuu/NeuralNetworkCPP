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

    nn::Matrix input(3, 1, {1.0, 2.0, -1.0});
    nn::Matrix output = sigmoid.forward(input);

    // Verify forward pass
    EXPECT_NEAR(output(0, 0), 0.73105, 1e-5);
    EXPECT_NEAR(output(1, 0), 0.88079, 1e-5);
    EXPECT_NEAR(output(2, 0), 0.26894, 1e-5);

    nn::Matrix gradOutput(3, 1, {0.1, 0.2, 0.3});
    nn::Matrix gradInput = sigmoid.backward(gradOutput);

    // Verify backward pass
    EXPECT_NEAR(gradInput(0, 0), 0.19661, 1e-5);
    EXPECT_NEAR(gradInput(1, 0), 0.10499, 1e-5);
    EXPECT_NEAR(gradInput(2, 0), 0.19661, 1e-5);
}

TEST(ActivationsTests, Softmax)
{
    nn::Softmax softmax;

    nn::Matrix input(3, 1, {1.0, 2.0, -1.0});
    nn::Matrix output = softmax.forward(input);

    // Verify forward pass
    EXPECT_NEAR(output(0, 0), 0.25949, 1e-5);
    EXPECT_NEAR(output(1, 0), 0.70538, 1e-5);
    EXPECT_NEAR(output(2, 0), 0.03511, 1e-5);

    nn::Matrix gradOutput(3, 1, {0.1, 0.2, 0.3});
    nn::Matrix gradInput = softmax.backward(gradOutput);

    // Verify backward pass
    EXPECT_NEAR(gradInput(0, 0), 0.19215, 1e-5);
    EXPECT_NEAR(gradInput(1, 0), 0.20781, 1e-5);
    EXPECT_NEAR(gradInput(2, 0), 0.03388, 1e-5);
}