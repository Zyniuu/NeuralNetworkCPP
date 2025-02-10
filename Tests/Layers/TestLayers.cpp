/**
 * C++ neural network library
 *
 * TestLayers.cpp
 */

#include <gtest/gtest.h>
#include <NeuralNetworkCPP/Layers/DenseLayer/DenseLayer.hpp>
#include <NeuralNetworkCPP/Optimizers/Adam/Adam.hpp>
#include <filesystem>

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
    nn::Matrix input(3, 1, {1.0, 2.0, 3.0});
    nn::Matrix output = layer.forward(input);

    nn::Matrix gradient(1, 1, {0.5});
    nn::Adam optimizer{};
    nn::Matrix gradInput = layer.backward(gradient, optimizer);

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