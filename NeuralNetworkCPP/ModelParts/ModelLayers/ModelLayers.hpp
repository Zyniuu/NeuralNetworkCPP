/**
 * C++ neural network library
 *
 * ModelLayers.hpp
 */

#include "../../Layers/Layers.hpp"

namespace nn
{
    class ModelLayers
    {
    protected:
        std::vector<std::unique_ptr<Layer>> m_layers; ///< Vector of layers in the network.

    public:
        /**
         * @brief Adds a layer to the network.
         *
         * @param layer The layer to add.
         */
        void addLayer(std::unique_ptr<Layer> layer);

    protected:
        /**
         * @brief Initializes a layer based on the provided layer type.
         *
         * @param layerType Enum value of the layer type.
         * @param file Input file stream (must be opened in binary mode).
         * @throws std::runtime_error If the layer type is invalid.
         */
        void initLayer(e_layerType layerType, std::ifstream &file);
    };
}