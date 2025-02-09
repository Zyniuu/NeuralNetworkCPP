/**
 * C++ neural network library
 *
 * NeuralNetworkCPP.cpp
 */

#include "NeuralNetworkCPP.hpp"
#include "GlobalThreadPool/GlobalThreadPool.hpp"

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
        int epochs, 
        int batchSize, 
        double validationSplit
    )
    {

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
}
