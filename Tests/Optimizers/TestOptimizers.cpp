/**
 * C++ neural network library
 *
 * TestOptimizers.cpp
 */

#include <gtest/gtest.h>
#include "../../NeuralNetworkCPP/Optimizers/SGD/SGD.hpp"
#include "../../NeuralNetworkCPP/Optimizers/RMSprop/RMSprop.hpp"
#include "../../NeuralNetworkCPP/Optimizers/Adam/Adam.hpp"

TEST(OptimizerTests, SGDWithMomentum)
{
    nn::SGD sgd(0.01);

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

TEST(OptimizerTests, RMSprop)
{
    nn::RMSprop rmsprop(0.01);

    nn::Matrix weights(2, 2, {1.0, 2.0, 3.0, 4.0});
    nn::Matrix biases(2, 1, {0.5, 0.5});
    nn::Matrix gradWeights(2, 2, {0.1, 0.2, 0.3, 0.4});
    nn::Matrix gradBiases(2, 1, {0.05, 0.05});

    rmsprop.update(weights, biases, gradWeights, gradBiases);

    // Verify updated weights and biases
    EXPECT_NEAR(weights(0, 0), 0.96837, 1e-5);
    EXPECT_NEAR(weights(0, 1), 1.96838, 1e-5);
    EXPECT_NEAR(weights(1, 0), 2.96838, 1e-5);
    EXPECT_NEAR(weights(1, 1), 3.96838, 1e-5);
    EXPECT_NEAR(biases(0, 0), 0.46837, 1e-5);
    EXPECT_NEAR(biases(1, 0), 0.46837, 1e-5);
}

TEST(OptimizerTests, Adam)
{
    nn::Adam adam(0.01);

    nn::Matrix weights(2, 2, {1.0, 2.0, 3.0, 4.0});
    nn::Matrix biases(2, 1, {0.5, 0.5});
    nn::Matrix gradWeights(2, 2, {0.1, 0.2, 0.3, 0.4});
    nn::Matrix gradBiases(2, 1, {0.05, 0.05});

    adam.update(weights, biases, gradWeights, gradBiases);

    // Verify updated weights and biases
    EXPECT_NEAR(weights(0, 0), 0.99, 1e-2);
    EXPECT_NEAR(weights(0, 1), 1.99, 1e-2);
    EXPECT_NEAR(weights(1, 0), 2.99, 1e-2);
    EXPECT_NEAR(weights(1, 1), 3.99, 1e-2);
    EXPECT_NEAR(biases(0, 0), 0.49, 1e-2);
    EXPECT_NEAR(biases(0, 1), 0.49, 1e-2);
}