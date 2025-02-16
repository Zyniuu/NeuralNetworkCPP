/**
 * C++ neural network library
 *
 * MNISTExample.cpp
 */

#include <NeuralNetworkCPP/NeuralNetworkCPP.hpp>
#include <NeuralNetworkCPP/Utils/Utils.hpp>
#include <NeuralNetworkCPP/DataPreprocessing/CSVReader/CSVReader.hpp>
#include <NeuralNetworkCPP/DataPreprocessing/Scalers/MinMaxScaler/MinMaxScaler.hpp>

int main()
{
    std::cout << "[    START ] Reading csv file data." << std::endl; 

    // Create a scaler
    nn::MinMaxScaler scaler;

    // Read training and testing data from csv files
    nn::CSVReader mnistTrain("mnist_train.csv", ',', false, true);
    nn::CSVReader mnistTest("mnist_test.csv", ',', false, true);
    mnistTrain.read();
    mnistTest.read();

    std::cout << "[     DONE ] Reading csv file data." << std::endl; 
    
    std::cout << "[    START ] Scaling data." << std::endl; 

    // Scale the data to the range of [0, 1]
    std::vector<std::vector<double>> trainData = scaler.fitTransform(mnistTrain.getData());

    // Convert labels into one-hot encoded vectors
    std::vector<std::vector<double>> trainLabels = nn::to_categorical(mnistTrain.getLabels());

    std::vector<std::vector<double>> testData = scaler.fitTransform(mnistTest.getData());
    std::vector<std::vector<double>> testLabels = nn::to_categorical(mnistTest.getLabels());

    std::cout << "[     DONE ] Scaling data." << std::endl << std::endl; 

    // Create a neural network model
    nn::NeuralNetworkCPP model;

    // Add layers to the model
    model.addLayer(std::make_unique<nn::DenseLayer>(28 * 28, 64, nn::HE_NORMAL, nn::RELU));
    model.addLayer(std::make_unique<nn::DenseLayer>(64, 10, nn::XAVIER_UNIFORM, nn::SOFTMAX));

    // Compile the model:
    // - Use the Adam optimizer with a learning rate of 0.003
    // - Use Categorical Cross-Entropy as the loss function
    // - Log out the accuracy computed on validation split
    model.compile(
        std::make_unique<nn::Adam>(0.003),
        std::make_unique<nn::CategoricalCrossEntropy>(),
        { nn::ACCURACY_LOG }
    );

    // Train the model:
    // - Training data: trainData (inputs) and trainLabels (targets)
    // - Number of epochs: 10
    // - Batch size: 512
    // - Validation split: 0.2 (20% of the training data)
    // - Patience: 1 (stop the training as soon as the network stops improving)
    // - minDelta: 0.00001 (this is minimum improvement of the error for the patience to reset)
    // - Verbose: Print the training progress
    model.train(trainData, trainLabels, 10, 512, 0.2, 1, 0.00001, true);

    // Evaluate the model's accuracy on the test data
    std::cout << "\nFinal accuracy: " << model.evaluate(testData, testLabels) << std::endl << std::endl;

    std::cout << "[    START ] Saving model to the file." << std::endl; 

    // Save the trained model to the file
    model.save("MNIST_model.bin");

    std::cout << "[     DONE ] Saving model to the file." << std::endl << std::endl; 

    return 0;
}