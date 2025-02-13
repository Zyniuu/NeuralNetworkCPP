/**
 * C++ neural network library
 *
 * TestLayers.cpp
 */

#include <gtest/gtest.h>
#include <NeuralNetworkCPP/Layers/Layers.hpp>
#include <NeuralNetworkCPP/Optimizers/Adam/Adam.hpp>
#include <filesystem>
#include <cmath>

TEST(DenseLayerTests, ForwardPass)
{
    nn::DenseLayer denseLayer(2, 3, nn::HE_NORMAL, nn::RELU);

    nn::Matrix input(2, 1, {1.0, 2.0});
    nn::Matrix output = denseLayer.forward(input);

    // Verify output dimensions
    EXPECT_EQ(output.getRows(), 3);
    EXPECT_EQ(output.getCols(), 1);

    // Verify that ReLU was applied (output should be non-negative)
    for (int i = 0; i < output.getRows(); i++)
    {
        for (int j = 0; j < output.getCols(); j++)
        {
            EXPECT_GE(output(i, j), 0.0);
        }
    }
}

TEST(DenseLayerTests, BackwardPass)
{
    nn::DenseLayer layer(3, 1, nn::HE_NORMAL, nn::RELU);
    layer.resetGradients();

    nn::Matrix input(3, 1, {1.0, 2.0, 3.0});
    nn::Matrix output = layer.forward(input);

    nn::Matrix gradient(1, 1, {0.5});
    nn::Matrix gradInput = layer.backward(gradient);

    // Check gradient dimensions
    ASSERT_EQ(gradInput.getRows(), 3);
    ASSERT_EQ(gradInput.getCols(), 1);
}

TEST(DenseLayerTests, SaveAndLoad)
{
    nn::DenseLayer layer(3, 2, nn::HE_NORMAL, nn::RELU);

    // Save the layer to a file
    std::ofstream outFile("layer.bin", std::ios::binary);
    layer.save(outFile);
    outFile.close();

    // Load the layer from the file
    std::ifstream inFile("layer.bin", std::ios::binary);
    nn::DenseLayer loadedLayer(inFile);
    inFile.close();

    // Verify that the loaded layer produces the same output
    nn::Matrix input(3, 1, {1.0, 2.0, 3.0});
    nn::Matrix originalOutput = layer.forward(input);
    nn::Matrix loadedOutput = loadedLayer.forward(input);

    ASSERT_EQ(originalOutput, loadedOutput);

    std::filesystem::remove("layer.bin");
}

TEST(BatchNormalizationTests, ForwardPass)
{
    nn::BatchNormalization bnLayer(3, 0.99, 1e-15);

    // Training mode
    bnLayer.setTrainingMode(true);

    nn::Matrix input(3, 1, {1.0, 2.0, 3.0});
    nn::Matrix output1 = bnLayer.forward(input);

    double mean = (1.0 + 2.0 + 3.0) / 3.0;
    double stddev = (std::pow(1.0 - mean, 2) + std::pow(2.0 - mean, 2) + std::pow(3.0 - mean, 2)) / 3.0;
    double runningMean = 0.99 * 0.0 + (1.0 - 0.99) * mean;
    double runningVar = 0.99 * 1.0 + (1.0 - 0.99) * stddev;
    nn::Matrix trainNorm = nn::Matrix(3, 1, {
        ((1.0 - mean) / std::sqrt(stddev + 1e-15)),
        ((2.0 - mean) / std::sqrt(stddev + 1e-15)),
        ((3.0 - mean) / std::sqrt(stddev + 1e-15))
    });

    EXPECT_EQ(output1, trainNorm);

    // Inference mode
    bnLayer.setTrainingMode(false);
    nn::Matrix output2 = bnLayer.forward(input);

    nn::Matrix inferNorm = nn::Matrix(3, 1, {
        ((1.0 - runningMean) / std::sqrt(runningVar + 1e-15)),
        ((2.0 - runningMean) / std::sqrt(runningVar + 1e-15)),
        ((3.0 - runningMean) / std::sqrt(runningVar + 1e-15))
    });

    EXPECT_EQ(output2, inferNorm);
}

TEST(BatchNormalizationTests, BackwardPass)
{
    nn::BatchNormalization bnLayer(3, 0.99, 1e-15);

    // Training mode
    bnLayer.setTrainingMode(true);

    nn::Matrix input(3, 1, {1.0, 2.0, 3.0});
    nn::Matrix grad(3, 1, {0.1, 0.2, 0.3});
    bnLayer.forward(input);
    nn::Matrix output = bnLayer.backward(grad);

    double mean = (1.0 + 2.0 + 3.0) / 3.0;
    double stddev = (std::pow(1.0 - mean, 2) + std::pow(2.0 - mean, 2) + std::pow(3.0 - mean, 2)) / 3.0;
    double t = 1 / std::sqrt(stddev + 1e-15);
    int m = 3;
    nn::Matrix trainNorm = nn::Matrix(3, 1, {
        ((1.0 - mean) / std::sqrt(stddev + 1e-15)),
        ((2.0 - mean) / std::sqrt(stddev + 1e-15)),
        ((3.0 - mean) / std::sqrt(stddev + 1e-15))
    });

    nn::Matrix inputGrad = (1.0 * t / m) * (m * grad - (t * t) * (input - mean) * grad.cwiseProduct(input - mean).sum() - grad.sum());

    EXPECT_EQ(output, inputGrad);
}

TEST(BatchNormalizationTests, SaveAndLoad)
{
    nn::BatchNormalization bnLayer(3, 0.99, 1e-15);

    // Training mode
    bnLayer.setTrainingMode(true);
    nn::Matrix input(3, 1, {1.0, 2.0, 3.0});

    // Save to file
    std::ofstream outFile("layer.bin", std::ios::binary);
    bnLayer.save(outFile);
    outFile.close();

    // Load from file
    std::ifstream inFile("layer.bin", std::ios::binary);
    nn::BatchNormalization bnLayer2(inFile);
    inFile.close();

    // Inference mode
    bnLayer.setTrainingMode(false);
    bnLayer2.setTrainingMode(false);

    nn::Matrix output = bnLayer.forward(input);
    nn::Matrix output2 = bnLayer2.forward(input);

    for (int i = 0; i < output.getRows(); i++)
        for (int j = 0; j < output.getCols(); j++)
            EXPECT_NEAR(output(i, j), output2(i, j), 1e-6);

    std::filesystem::remove("layer.bin");
}