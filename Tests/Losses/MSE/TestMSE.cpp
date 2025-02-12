/**
 * C++ neural network library
 *
 * TestMSE.cpp
 */

#include <gtest/gtest.h>
#include <NeuralNetworkCPP/Losses/MeanSquaredError/MeanSquaredError.hpp>

// Test if MSE computes loss correctly
TEST(MSETests, ComputeLoss)
{
    nn::MeanSquaredError mse;

    // Test case 1: Simple 1x1 matrix
    nn::Matrix predictions1(1, 1, {2.0});
    nn::Matrix targets1(1, 1, {3.0});
    EXPECT_DOUBLE_EQ(mse.computeLoss(predictions1, targets1), 1.0); // (3-2)^2 = 1

    // Test case 2: 2x2 matrix
    nn::Matrix predictions2(2, 2, {1.0, 2.0, 3.0, 4.0});
    nn::Matrix targets2(2, 2, {1.0, 3.0, 2.0, 5.0});
    double expectedLoss2 = (0.0 + 1.0 + 1.0 + 1.0) / 4;
    EXPECT_DOUBLE_EQ(mse.computeLoss(predictions2, targets2), expectedLoss2);

    // Test case 3: Invalid dimensions
    nn::Matrix predictions3(2, 2);
    nn::Matrix targets3(1, 2);
    EXPECT_THROW(mse.computeLoss(predictions3, targets3), std::invalid_argument);
}

// Test if MSE computes gradient correctly
TEST(MSETests, ComputeGradient)
{
    nn::MeanSquaredError mse;

    // Test case 1: Simple 1x1 matrix
    nn::Matrix predictions1(1, 1, {2.0});
    nn::Matrix targets1(1, 1, {3.0});
    nn::Matrix gradient1 = mse.computeGradient(predictions1, targets1);
    EXPECT_DOUBLE_EQ(gradient1(0, 0), 2 * (2.0 - 3.0) / 1); // 2*(2-3)/1 = -2

    // Test case 2: 2x2 matrix
    nn::Matrix predictions2(2, 2, {1.0, 2.0, 3.0, 4.0});
    nn::Matrix targets2(2, 2, {1.0, 3.0, 2.0, 5.0});
    nn::Matrix gradient2 = mse.computeGradient(predictions2, targets2);
    EXPECT_DOUBLE_EQ(gradient2(0, 0), 2 * (1.0 - 1.0) / 4); // 0
    EXPECT_DOUBLE_EQ(gradient2(0, 1), 2 * (2.0 - 3.0) / 4); // -0.5
    EXPECT_DOUBLE_EQ(gradient2(1, 0), 2 * (3.0 - 2.0) / 4); // 0.5
    EXPECT_DOUBLE_EQ(gradient2(1, 1), 2 * (4.0 - 5.0) / 4); // -0.5

    // Test case 3: Invalid dimensions
    nn::Matrix predictions3(2, 2);
    nn::Matrix targets3(1, 2);
    EXPECT_THROW(mse.computeGradient(predictions3, targets3), std::invalid_argument);
}