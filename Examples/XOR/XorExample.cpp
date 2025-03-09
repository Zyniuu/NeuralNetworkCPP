/**
 * C++ neural network library
 *
 * XorExample.cpp
 */

#include <NeuralNetworkCPP/NeuralNetworkCPP.hpp>
#include <cmath>

int main()
{
    // Define the input data (X) and target labels (Y) for the XOR problem
    std::vector<std::vector<double>> xData = {
        {0.0, 0.0}, // Input 1: [0, 0]
        {0.0, 1.0}, // Input 2: [0, 1]
        {1.0, 0.0}, // Input 3: [1, 0]
        {1.0, 1.0}  // Input 4: [1, 1]
    };

    std::vector<std::vector<double>> yData = {
        {0.0}, // Output for [0, 0] -> 0
        {1.0}, // Output for [0, 1] -> 1
        {1.0}, // Output for [1, 0] -> 1
        {0.0}  // Output for [1, 1] -> 0
    };

    // Create a neural network model
    nn::NeuralNetworkCPP model;

    // Add layers to the model
    model.addLayer(std::make_unique<nn::DenseLayer>(2, 8, nn::HE_NORMAL, nn::RELU));
    model.addLayer(std::make_unique<nn::DenseLayer>(8, 1, nn::XAVIER_UNIFORM, nn::SIGMOID));

    // Compile the model:
    // - Use the Adam optimizer with a learning rate of 0.01
    // - Use Binary Cross-Entropy as the loss function
    model.compile(
        std::make_unique<nn::Adam>(0.01),
        std::make_unique<nn::BinaryCrossEntropy>()
    );

    // Train the model:
    // - Training data: xData (inputs) and yData (targets)
    // - Number of epochs: 1000
    // - Batch size: 4 (since there are only 4 samples)
    // - Validation split: 0.0 (no validation data)
    // - Patience: 10 (wait for max 10 epochs for the network to improve)
    // - minDelta: 0.00001 (this is minimum improvement of the error for the patience to reset)
    // - Verbose: Print the training progress
    model.train(xData, yData, 1000, 4, 0.0, 10, 0.00001, true);

    // Make predictions using the trained model
    std::cout << "\nPredictions:" << std::endl;
    std::cout << "[0, 0] -> " << std::round(model.predict({0.0, 0.0})[0]) << std::endl;
    std::cout << "[0, 1] -> " << std::round(model.predict({0.0, 1.0})[0]) << std::endl;
    std::cout << "[1, 0] -> " << std::round(model.predict({1.0, 0.0})[0]) << std::endl;
    std::cout << "[1, 1] -> " << std::round(model.predict({1.0, 1.0})[0]) << std::endl << std::endl;

    // Evaluate the model's accuracy on the training data
    std::cout << "Final accuracy: " << model.evaluate(xData, yData) << std::endl;

    return 0;
}