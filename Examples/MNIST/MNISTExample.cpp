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
    nn::MinMaxScaler scaler;
    nn::CSVReader mnistTrain("mnist_train.csv", ',', false, true);
    nn::CSVReader mnistTest("mnist_test.csv", ',', false, true);
    mnistTrain.read();
    mnistTest.read();

    nn::Matrix tempX(mnistTrain.getData()[0].size(), 1, mnistTrain.getData()[0]);
    nn::Matrix tempY(mnistTrain.getLabels()[0].size(), 1, mnistTrain.getLabels()[0]);

    std::vector<std::vector<double>> trainData = scaler.fitTransform(mnistTrain.getData());
    std::vector<std::vector<double>> trainLabels = nn::to_categorical(mnistTrain.getLabels());

    std::vector<std::vector<double>> testData = scaler.fitTransform(mnistTest.getData());
    std::vector<std::vector<double>> testLabels = nn::to_categorical(mnistTest.getLabels());

    nn::NeuralNetworkCPP model;
    model.addLayer(std::make_unique<nn::DenseLayer>(28 * 28, 64, nn::HE_NORMAL, nn::RELU));
    model.addLayer(std::make_unique<nn::DenseLayer>(64, 10, nn::XAVIER_UNIFORM, nn::SOFTMAX));

    model.compile(
        std::make_unique<nn::Adam>(),
        std::make_unique<nn::CategoricalCrossEntropy>(),
        { nn::ACCURACY_LOG }
    );

    model.train(trainData, trainLabels, 10, 512, 0.2, 1, 0.00001, true);

    std::cout << "\nFinal accuracy: " << model.evaluate(testData, testLabels) << std::endl;

    return 0;
}