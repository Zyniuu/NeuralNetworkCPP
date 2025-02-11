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
        // Initialize the global thread pool with the specified number of threads
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
        // Convert the input vector to a matrix
        Matrix output = Matrix(input.size(), 1, input);

        // Perform forward propagation
        output = forward(output);

        // Return the output as a vector
        return output.getData();
    }

    double NeuralNetworkCPP::evaluate(const std::vector<std::vector<double>> &xTest, const std::vector<std::vector<double>> &yTest)
    {
        // Check if the test data is empty
        if (xTest.empty() || yTest.empty())
            return 0.0;

        int correct = 0;

        // Iterate over the test data
        for (int i = 0; i < xTest.size(); i++)
        {
            // Predict the output for the current input
            std::vector<double> output = predict(xTest[i]);

            // If classification (multiple outputs)
            if (output.size() > 1)
            {
                // Find the index of the maximum output
                int maxOutputIndex = std::max_element(output.begin(), output.end()) - output.begin();
                int maxTargetIndex = std::max_element(yTest[i].begin(), yTest[i].end()) - yTest[i].begin();

                // Check if the prediction is correct
                if (maxOutputIndex == maxTargetIndex)
                    correct++;
            }
            // If regression (single output)
            else if (std::round(output[0]) == std::round(yTest[i][0]))
                correct++;
        }

        // Return the accuracy
        return static_cast<double>(correct) / xTest.size();
    }

    void NeuralNetworkCPP::addLayer(std::unique_ptr<Layer> layer)
    {
        // Add the layer to the network
        m_layers.push_back(std::move(layer));
    }

    void NeuralNetworkCPP::compile(std::unique_ptr<Optimizer> optimizer, std::unique_ptr<Loss> lossFunc, std::vector<e_metric> metrics)
    {
        // Set the optimizer, loss function, and logger
        m_optimizer = std::move(optimizer);
        m_loss = std::move(lossFunc);
        m_logger = std::make_unique<Logger>(metrics);
    }

    void NeuralNetworkCPP::train(
        const std::vector<std::vector<double>> &xTrain, 
        const std::vector<std::vector<double>> &yTrain, 
        const int epochs, 
        const int batchSize, 
        const double validationSplit,
        const bool verbose
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

        // Compute total number of batches
        double totalBatches = static_cast<double>(xTrainSplit.size()) / static_cast<double>(batchSize);

        // Log training start
        if (verbose)
            m_logger->logTrainingStart();

        // Training loop
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            // Log epoch start
            if (verbose)
                m_logger->logEpochStart(epoch + 1, epochs);

            int batchIndex = 0;
            double loss = 0.0;

            // Shuffle training data before each epoch
            shuffleDataset(xTrainSplit, yTrainSplit);

            // Process batches
            for (int i = 0; i < xTrainSplit.size(); i += batchSize)
            {
                batchIndex++;

                // Log batch progress
                if (verbose)
                    m_logger->logBatch(batchIndex, std::ceil(totalBatches));

                // Make sure batch doesn't overflow
                int end = std::min(static_cast<int>(xTrainSplit.size() - 1), i + batchSize);

                // Get the current batch
                std::vector<std::vector<double>> xBatch = slice(xTrainSplit, i, end);
                std::vector<std::vector<double>> yBatch = slice(yTrainSplit, i, end);

                // Train on the current batch
                trainOnBatch(xBatch, yBatch, loss);
            }

            // Compute average loss and accuracy
            loss /= xTrainSplit.size();
            double accuracy = evaluate(xValSplit, yValSplit);

            // Log epoch end
            if (verbose)
                m_logger->logEpochEnd(epochs, loss, accuracy);
        }

        // Log training end
        if (verbose)
            m_logger->logTrainingEnd();
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
            grad = (*it)->backward(grad);
        }
    }

    void NeuralNetworkCPP::initLayer(e_layerType layerType, std::ifstream &file)
    {
        // Initialize the layer based on the type
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
        // Reset gradients for all layers
        for (auto &layer : m_layers)
            layer->resetGradients();

        // Forward and backward passes (accumulating gradients)
        for (int i = 0; i < xBatch.size(); i++)
        {
            // Convert vectors to matrices
            Matrix input = Matrix(xBatch[i].size(), 1, xBatch[i]);
            Matrix target = Matrix(yBatch[i].size(), 1, yBatch[i]);

            // Forward pass
            Matrix output = forward(input);
            loss += m_loss->computeLoss(output, target);

            // Backward pass
            Matrix grad = m_loss->computeGradient(output, target);
            backward(grad);
        }

        // Average gradients and update weights
        for (auto &layer : m_layers)
            layer->applyGradient(*m_optimizer, xBatch.size());
    }
}
