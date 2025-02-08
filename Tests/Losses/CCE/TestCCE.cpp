/**
 * C++ neural network library
 *
 * TestCCE.cpp
 */

#include <gtest/gtest.h>
#include <cmath>
#include <NeuralNetworkCPP/Losses/CategoricalCrossEntropy/CategoricalCrossEntropy.hpp>

// Test if CCE computes loss correctly
TEST(CCETests, ComputeLoss)
{
    nn::CategoricalCrossEntropy cce;

    // Test case 1: Simple 1x2 matrix (one-hot encoded)
    nn::Matrix predictions1(1, 2, {0.7, 0.3});
    nn::Matrix targets1(1, 2, {1.0, 0.0});
    double expectedLoss1 = -std::log(0.7 + 1e-15);
    EXPECT_DOUBLE_EQ(cce.computeLoss(predictions1, targets1), expectedLoss1);

    // Test case 2: 2x3 matrix (one-hot encoded)
    nn::Matrix predictions2(2, 3, {0.1, 0.7, 0.2, 0.4, 0.5, 0.1});
    nn::Matrix targets2(2, 3, {0.0, 1.0, 0.0, 1.0, 0.0, 0.0});
    double expectedLoss2 = (-std::log(0.7 + 1e-15) - std::log(0.4 + 1e-15)) / 2;
    EXPECT_DOUBLE_EQ(cce.computeLoss(predictions2, targets2), expectedLoss2);

    // Test case 3: Invalid dimensions
    nn::Matrix predictions3(2, 2);
    nn::Matrix targets3(1, 2);
    EXPECT_THROW(cce.computeLoss(predictions3, targets3), std::invalid_argument);
}

TEST(CCETests, ComputeGradient)
{
    nn::CategoricalCrossEntropy cce;

    // Test case 1: Simple 1x2 matrix (one-hot encoded)
    nn::Matrix predictions1(1, 2, {0.7, 0.3});
    nn::Matrix targets1(1, 2, {1.0, 0.0});
    nn::Matrix gradient1 = cce.computeGradient(predictions1, targets1);
    EXPECT_DOUBLE_EQ(gradient1(0, 0), -1.0 / (0.7 + 1e-15));
    EXPECT_DOUBLE_EQ(gradient1(0, 1), 0.0); // 0 (since target is 0)

    // Test case 2: 2x3 matrix (one-hot encoded)
    nn::Matrix predictions2(2, 3, {0.1, 0.7, 0.2, 0.4, 0.5, 0.1});
    nn::Matrix targets2(2, 3, {0.0, 1.0, 0.0, 1.0, 0.0, 0.0});
    nn::Matrix gradient2 = cce.computeGradient(predictions2, targets2);
    EXPECT_DOUBLE_EQ(gradient2(0, 0), 0.0);                  // 0 (since target is 0)
    EXPECT_DOUBLE_EQ(gradient2(0, 1), -1.0 / (0.7 + 1e-15)); // -1/0.7
    EXPECT_DOUBLE_EQ(gradient2(0, 2), 0.0);                  // 0 (since target is 0)
    EXPECT_DOUBLE_EQ(gradient2(1, 0), -1.0 / (0.4 + 1e-15)); // -1/0.4
    EXPECT_DOUBLE_EQ(gradient2(1, 1), 0.0);                  // 0 (since target is 0)
    EXPECT_DOUBLE_EQ(gradient2(1, 2), 0.0);                  // 0 (since target is 0)

    // Test case 3: Invalid dimensions
    nn::Matrix predictions3(2, 2);
    nn::Matrix targets3(1, 2);
    EXPECT_THROW(cce.computeGradient(predictions3, targets3), std::invalid_argument);
}