/**
 * C++ neural network library
 *
 * BostonHousingExample.cpp
 */

#include <NeuralNetworkCPP/NeuralNetworkCPP.hpp>
#include <NeuralNetworkCPP/DataPreprocessing/CSVReader/CSVReader.hpp>
#include <NeuralNetworkCPP/DataPreprocessing/Scalers/StandardScaler/StandardScaler.hpp>

int main()
{
    std::cout << "[    START ] Reading csv file data." << std::endl; 

    // Create a scaler
    nn::StandardScaler scaler;

    // Read training and testing data from csv files
    nn::CSVReader bostonTrain("BostonHousing_train.csv", ',', true, true);
    nn::CSVReader bostonTest("BostonHousing_test.csv", ',', true, true);
    bostonTrain.read();
    bostonTest.read();

    std::cout << "[     DONE ] Reading csv file data." << std::endl;

    std::cout << "[    START ] Scaling data." << std::endl; 

    // Scale the data
    std::vector<std::vector<double>> trainData = scaler.fitTransform(bostonTrain.getData());
    std::vector<std::vector<double>> testData = scaler.transform(bostonTest.getData());

    std::cout << "[     DONE ] Scaling data." << std::endl << std::endl;

    // Create a neural network model
    nn::NeuralNetworkCPP model;

    // Add layers to the model
    model.addLayer(std::make_unique<nn::DenseLayer>(13, 26, nn::HE_NORMAL, nn::RELU));
    model.addLayer(std::make_unique<nn::DenseLayer>(26, 26, nn::HE_NORMAL, nn::RELU));
    model.addLayer(std::make_unique<nn::DenseLayer>(26, 1, nn::XAVIER_UNIFORM, nn::NONE));

    // Compile the model:
    // - Use the Adam optimizer with default learning rate (0.001)
    // - Use Mean Squared Error as the loss function
    // - Log out the Mean Absolute Error computed on validation split
    model.compile(
        std::make_unique<nn::Adam>(),
        std::make_unique<nn::MeanSquaredError>(),
        { nn::MAE_LOG }
    );

    // Train the model:
    // - Training data: trainData (inputs) and bostonTrain labels (targets)
    // - Number of epochs: 20
    // - Batch size: 1
    // - Validation split: 0.2 (20% of the training data)
    // - Patience: 1 (stop the training as soon as the network stops improving)
    // - minDelta: 0.00001 (this is minimum improvement of the error for the patience to reset)
    // - Verbose: Print the training progress
    model.train(trainData, bostonTrain.getLabels(), 20, 1, 0.2, 1, 0.00001, true);

    // Evaluate the model's mae on the test data
    std::cout << "\nFinal MAE: " << model.evaluate(testData, bostonTest.getLabels(), nn::MAE_LOG) << std::endl << std::endl;

    std::cout << "[    START ] Saving model to the file." << std::endl; 

    // Save the trained model to the file
    model.save("BostonHousing_model.bin");

    std::cout << "[     DONE ] Saving model to the file." << std::endl << std::endl; 

    return 0;
}