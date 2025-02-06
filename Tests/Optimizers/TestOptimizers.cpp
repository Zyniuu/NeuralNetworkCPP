/**
 * C++ neural network library
 *
 * TestOptimizers.cpp
 */

#include <gtest/gtest.h>
#include "../../NeuralNetworkCPP/Optimizers/SGD/SGD.hpp"

TEST(OptimizerTests, SGDWithMomentum)
{
    nn::SGD sgd(0.01, 0.9);

    nn::Matrix weights(2, 2, {1.0, 2.0, 3.0, 4.0});
    nn::Matrix biases(2, 1, {0.5, 0.5});
    nn::Matrix gradWeights(2, 2, {0.1, 0.2, 0.3, 0.4});
    nn::Matrix gradBiases(2, 1, {0.05, 0.05});

    sgd.update(weights, biases, gradWeights, gradBiases);

    // Verify updated weights and biases
    EXPECT_NEAR(weights(0, 0), 0.999, 1e-3);
    EXPECT_NEAR(weights(0, 1), 1.998, 1e-3);
    EXPECT_NEAR(weights(1, 0), 2.997, 1e-3);
    EXPECT_NEAR(weights(1, 1), 3.996, 1e-3);
    EXPECT_NEAR(biases(0, 0), 0.4995, 1e-3);
    EXPECT_NEAR(biases(1, 0), 0.4995, 1e-3);
}