/**
 * C++ neural network library
 *
 * ModelLayers.cpp
 */

#include "ModelLayers.hpp"

namespace nn
{
    void ModelLayers::addLayer(std::unique_ptr<Layer> layer)
    {
        // Add the layer to the network
        m_layers.push_back(std::move(layer));
    }

    void ModelLayers::initLayer(e_layerType layerType, std::ifstream &file)
    {
        // Initialize the layer based on the type
        switch (layerType)
        {
        case DENSE:
            addLayer(std::make_unique<DenseLayer>(file));
            break;
        
        case BATCH_NORM:
            addLayer(std::make_unique<BatchNormalization>(file));
            break;
        
        default:
            throw std::runtime_error("Invalid layer type.");
        }
    }
}