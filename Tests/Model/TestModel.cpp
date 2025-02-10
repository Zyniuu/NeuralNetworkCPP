/**
 * C++ neural network library
 *
 * TestModel.cpp
 */

#include <gtest/gtest.h>
#include <NeuralNetworkCPP/NeuralNetworkCPP.hpp>
#include <filesystem>
#include <cmath>

TEST(ModelTests, Predict)
{
    nn::NeuralNetworkCPP model;
    model.addLayer(std::make_unique<nn::DenseLayer>(2, 3, nn::HE_NORMAL, nn::RELU));
    model.addLayer(std::make_unique<nn::DenseLayer>(3, 1, nn::XAVIER_UNIFORM, nn::SIGMOID));

    std::vector<double> input = {1.0, 2.0};
    std::vector<double> output = model.predict(input);

    ASSERT_EQ(output.size(), 1);
}

TEST(ModelTests, Evaluate)
{
    nn::NeuralNetworkCPP model;
    model.addLayer(std::make_unique<nn::DenseLayer>(2, 3, nn::HE_NORMAL, nn::RELU));
    model.addLayer(std::make_unique<nn::DenseLayer>(3, 1, nn::XAVIER_UNIFORM, nn::SIGMOID));

    std::vector<std::vector<double>> xTest = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<std::vector<double>> yTest = {{0.0}, {1.0}};

    double accuracy = model.evaluate(xTest, yTest);
    ASSERT_GE(accuracy, 0.0);
    ASSERT_LE(accuracy, 1.0);
}

TEST(ModelTests, SaveAndLoad)
{
    nn::NeuralNetworkCPP model;
    model.addLayer(std::make_unique<nn::DenseLayer>(2, 3, nn::HE_NORMAL, nn::RELU));
    model.addLayer(std::make_unique<nn::DenseLayer>(3, 1, nn::XAVIER_UNIFORM, nn::SIGMOID));

    // Save the model
    model.save("test_model.bin");

    // Load the model
    nn::NeuralNetworkCPP loadedModel("test_model.bin");

    // Verify that the loaded model produces the same output
    std::vector<double> input = {1.0, 2.0};
    std::vector<double> output1 = model.predict(input);
    std::vector<double> output2 = loadedModel.predict(input);

    ASSERT_EQ(output1, output2);

    std::filesystem::remove("test_model.bin");
}

TEST(ModelTests, Train)
{
    nn::NeuralNetworkCPP model;
    model.addLayer(std::make_unique<nn::DenseLayer>(2, 4, nn::HE_NORMAL, nn::RELU));
    model.addLayer(std::make_unique<nn::DenseLayer>(4, 1, nn::XAVIER_UNIFORM, nn::SIGMOID));

    model.compile(
        std::make_unique<nn::SGD>(),
        std::make_unique<nn::MeanSquaredError>()
    );

    std::vector<std::vector<double>> xData = {
        {0.0, 0.0}, 
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    std::vector<std::vector<double>> yData = {
        {0.0}, 
        {1.0},
        {1.0},
        {0.0}
    };

    model.train(xData, yData, 10, 4, 0.0, true);

    EXPECT_EQ(std::round(model.predict({0.0, 0.0})[0]), 0);
    EXPECT_EQ(std::round(model.predict({0.0, 1.0})[0]), 1);
    EXPECT_EQ(std::round(model.predict({1.0, 0.0})[0]), 1);
    EXPECT_EQ(std::round(model.predict({1.0, 1.0})[0]), 0);
}