/**
 * C++ neural network library
 *
 * TestBCE.cpp
 */

#include <gtest/gtest.h>
#include <cmath>
#include "../../../NeuralNetworkCPP/Losses/BinaryCrossEntropy/BinaryCrossEntropy.hpp"

// Test if BCE computes loss correctly
TEST(BCETests, ComputeLoss)
{
    nn::BinaryCrossEntropy bce;

    // Test case 1: Simple 1x1 matrix
    nn::Matrix predictions1(1, 1, {0.8});
    nn::Matrix targets1(1, 1, {1.0});
    double expectedLoss1 = -std::log(0.8 + 1e-15); // -log(0.8)
    EXPECT_DOUBLE_EQ(bce.computeLoss(predictions1, targets1), expectedLoss1);

    // Test case 2: 2x2 matrix
    nn::Matrix predictions2(2, 2, {0.9, 0.1, 0.4, 0.6});
    nn::Matrix targets2(2, 2, {1.0, 0.0, 1.0, 0.0});
    double expectedLoss2 = (-std::log(0.9 + 1e-15) - std::log(1 - 0.1 + 1e-15) - std::log(0.4 + 1e-15) - std::log(1 - 0.6 + 1e-15)) / 2;
    EXPECT_DOUBLE_EQ(bce.computeLoss(predictions2, targets2), expectedLoss2);

    // Test case 3: Invalid dimensions
    nn::Matrix predictions3(2, 2);
    nn::Matrix targets3(1, 2);
    EXPECT_THROW(bce.computeLoss(predictions3, targets3), std::invalid_argument);
}

// Test if BCE computes gradient correctly
TEST(BCETests, ComputeGradient)
{
    nn::BinaryCrossEntropy bce;

    // Test case 1: Simple 1x1 matrix
    nn::Matrix predictions1(1, 1, {0.8});
    nn::Matrix targets1(1, 1, {1.0});
    nn::Matrix gradient1 = bce.computeGradient(predictions1, targets1);
    EXPECT_DOUBLE_EQ(gradient1(0, 0), -1.0 / (0.8 + 1e-15)); // -1/0.8

    // Test case 2: 2x2 matrix
    nn::Matrix predictions2(2, 2, {0.9, 0.1, 0.4, 0.6});
    nn::Matrix targets2(2, 2, {1.0, 0.0, 1.0, 0.0});
    nn::Matrix gradient2 = bce.computeGradient(predictions2, targets2);
    EXPECT_DOUBLE_EQ(gradient2(0, 0), -1.0 / (0.9 + 1e-15));       // -1/0.9
    EXPECT_DOUBLE_EQ(gradient2(0, 1), 1.0 / (1 - 0.1 + 1e-15));  // 1/(1-0.1)
    EXPECT_DOUBLE_EQ(gradient2(1, 0), -1.0 / (0.4 + 1e-15));       // -1/0.4
    EXPECT_DOUBLE_EQ(gradient2(1, 1), 1.0 / (1 - 0.6 + 1e-15));  // 1/(1-0.6)

    // Test case 3: Invalid dimensions
    nn::Matrix predictions3(2, 2);
    nn::Matrix targets3(1, 2);
    EXPECT_THROW(bce.computeGradient(predictions3, targets3), std::invalid_argument);
}