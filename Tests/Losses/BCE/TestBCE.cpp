/**
 * C++ neural network library
 *
 * TestBCE.cpp
 */

#include <gtest/gtest.h>
#include <cmath>
#include <NeuralNetworkCPP/Losses/BinaryCrossEntropy/BinaryCrossEntropy.hpp>

// Test if BCE computes loss correctly
TEST(BCETests, ComputeLoss)
{
    nn::BinaryCrossEntropy bce;

    // Test case 1: Simple 2x1 matrix
    nn::Matrix predictions1(2, 1, {0.7, 0.3});
    nn::Matrix targets1(2, 1, {1.0, 0.0});
    EXPECT_NEAR(bce.computeLoss(predictions1, targets1), 0.71334, 1e-5);

    // Test case 2: 2x2 matrix
    nn::Matrix predictions2(2, 3, {0.1, 0.7, 0.2, 0.4, 0.5, 0.1});
    nn::Matrix targets2(2, 3, {0.0, 1.0, 0.0, 1.0, 0.0, 0.0});
    EXPECT_NEAR(bce.computeLoss(predictions2, targets2), 0.79999, 1e-5);

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