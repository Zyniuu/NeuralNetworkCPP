/**
 * C++ neural network library
 *
 * NeuralNetworkCPP.cpp
 */

#include "NeuralNetworkCPP.hpp"
#include "GlobalThreadPool/GlobalThreadPool.hpp"
#include "Utils/Utils.hpp"
#include <cmath>

namespace nn
{
    NeuralNetworkCPP::NeuralNetworkCPP(const int numThreads)
    {
        initGlobalThreadPool(numThreads);
    }

    NeuralNetworkCPP::NeuralNetworkCPP(const std::string &filename, const int numThreads)
    {
        // Create a file stream
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("Failed to open file for reading.");
        
        // Read the number of layers
        int numLayers;
        file.read(reinterpret_cast<char *>(&numLayers), sizeof(numLayers));

        // Read each layer
        for (int i = 0; i < numLayers; i++)
        {
            // Read the layer type
            e_layerType layerType;
            file.read(reinterpret_cast<char *>(&layerType), sizeof(layerType));

            // Instantiate the correct layer type
            initLayer(layerType, file);
        }

        // Check if reading was successful
        if (!file.good())
            throw std::runtime_error("Failed to read model from the file.");

        // Initialize the global thread pool
        initGlobalThreadPool(numThreads);
    }

    std::vector<double> NeuralNetworkCPP::predict(const std::vector<double> &input)
    {
        Matrix output = Matrix(1, input.size(), input);
        output = forward(output);
        return output.getData();
    }

    double NeuralNetworkCPP::evaluate(const std::vector<std::vector<double>> &xTest, const std::vector<std::vector<double>> &yTest)
    {
        if (xTest.empty() || yTest.empty())
            return 0.0;

        int correct = 0;

        for (int i = 0; i < xTest.size(); i++)
        {
            std::vector<double> output = predict(xTest[i]);

            // If classification
            if (output.size() > 1)
            {
                int maxOutputIndex = std::max_element(output.begin(), output.end()) - output.begin();
                int maxTargetIndex = std::max_element(yTest[i].begin(), yTest[i].end()) - yTest[i].begin();

                if (maxOutputIndex == maxTargetIndex)
                    correct++;
            }
            else if (std::round(output[0]) == std::round(yTest[i][0]))
                correct++;
        }

        return static_cast<double>(correct) / xTest.size();
    }

    void NeuralNetworkCPP::addLayer(std::unique_ptr<Layer> layer)
    {
        m_layers.push_back(std::move(layer));
    }

    void NeuralNetworkCPP::compile(std::unique_ptr<Optimizer> optimizer, std::unique_ptr<Loss> lossFunc)
    {
        m_optimizer = std::move(optimizer);
        m_loss = std::move(lossFunc);
    }

    void NeuralNetworkCPP::train(
        const std::vector<std::vector<double>> &xTrain, 
        const std::vector<std::vector<double>> &yTrain, 
        const int epochs, 
        const int batchSize, 
        const double validationSplit
    )
    {
        // Copy the dataset
        std::vector<std::vector<double>> xTrainCopy = xTrain;
        std::vector<std::vector<double>> yTrainCopy = yTrain;

        // Shuffle the dataset
        shuffleDataset(xTrainCopy, yTrainCopy);

        // Split data into training and validation sets
        int numValidation = xTrainCopy.size() * validationSplit;
        std::vector<std::vector<double>> xValSplit = slice(xTrainCopy, 0, numValidation);
        std::vector<std::vector<double>> yValSplit = slice(yTrainCopy, 0, numValidation);
        std::vector<std::vector<double>> xTrainSplit = slice(xTrainCopy, numValidation, xTrainCopy.size());
        std::vector<std::vector<double>> yTrainSplit = slice(yTrainCopy, numValidation, yTrainCopy.size());

        // Training loop
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double loss = 0.0;

            // Shuffle training data before each epoch
            shuffleDataset(xTrainSplit, yTrainSplit);

            // Process batches
            for (int i = 0; i < xTrainSplit.size(); i += batchSize)
            {
                // Make sure batch doesn't overflow
                int end = std::min(static_cast<int>(xTrainSplit.size() - 1), i + batchSize);

                std::vector<std::vector<double>> xBatch = slice(xTrainSplit, i, end);
                std::vector<std::vector<double>> yBatch = slice(yTrainSplit, i, end);

                // Mini batch training
                trainOnBatch(xBatch, yBatch, loss);
            }

            loss /= xTrainSplit.size();
            double accuracy = evaluate(xValSplit, yValSplit);
        }
    }

    void NeuralNetworkCPP::save(const std::string &filename) const
    {
        // Create the file
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("Failed to open file for writing.");
        
        // Save the number of network layers
        int numLayers = m_layers.size();
        file.write(reinterpret_cast<const char *>(&numLayers), sizeof(numLayers));

        // Save each layer
        for (const auto &layer : m_layers)
        {
            // Save the layer type
            e_layerType layerType = layer->getType();
            file.write(reinterpret_cast<const char *>(&layerType), sizeof(layerType));

            // Save the layer
            layer->save(file);
        }

        // Check if writing was successful
        if (!file.good())
            throw std::runtime_error("Failed to write model to the file.");
    }

    Matrix NeuralNetworkCPP::forward(const Matrix &input)
    {
        // Propagate the input forward through the layers
        Matrix output = input;
        for (const auto &layer : m_layers)
        {
            output = layer->forward(output);
        }
        return output;
    }

    void NeuralNetworkCPP::backward(const Matrix &gradient)
    {
        // Propagate the gradient backward through the layers
        Matrix grad = gradient;
        for (auto it = m_layers.rbegin(); it != m_layers.rend(); it++)
        {
            grad = (*it)->backward(grad, *m_optimizer);
        }
    }

    void NeuralNetworkCPP::initLayer(e_layerType layerType, std::ifstream &file)
    {
        switch (layerType)
        {
        case DENSE:
            m_layers.push_back(std::make_unique<DenseLayer>(file));
            break;
        
        default:
            throw std::runtime_error("Invalid layer type.");
        }
    }

    void NeuralNetworkCPP::trainOnBatch(
        const std::vector<std::vector<double>> &xBatch, 
        const std::vector<std::vector<double>> &yBatch, 
        double &loss
    )
    {
        for (int i = 0; i < xBatch.size(); i++)
        {
            // Convert vectors to matrices
            Matrix input = Matrix(1, xBatch[i].size(), xBatch[i]);
            Matrix target = Matrix(1, yBatch[i].size(), yBatch[i]);

            // Forward pass
            Matrix output = forward(input);

            // Compute loss
            loss += m_loss->computeLoss(output, target);

            // Backward pass
            Matrix grad = m_loss->computeGradient(output, target);
            backward(grad);
        }
    }
}
