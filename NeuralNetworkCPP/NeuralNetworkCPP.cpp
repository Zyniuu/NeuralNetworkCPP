/**
 * C++ neural network library
 *
 * NeuralNetworkCPP.cpp
 */

#include "NeuralNetworkCPP.hpp"
#include "GlobalThreadPool/GlobalThreadPool.hpp"

namespace nn
{
    NeuralNetworkCPP::NeuralNetworkCPP(int numThreads)
    {
        initGlobalThreadPool(numThreads);
    }

    NeuralNetworkCPP::NeuralNetworkCPP(const std::string &filename, int numThreads)
    {
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

    }

    Matrix NeuralNetworkCPP::forward(const Matrix &input)
    {

    }

    void NeuralNetworkCPP::backward(const Matrix &gradient)
    {
        
    }
}