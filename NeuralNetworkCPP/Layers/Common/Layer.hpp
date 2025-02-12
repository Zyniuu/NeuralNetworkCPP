/**
 * C++ neural network library
 *
 * Layer.hpp
 */

#ifndef LAYER_HPP
#define LAYER_HPP

#include "../../Matrix/Matrix.hpp"
#include "../../Optimizers/Common/Optimizer.hpp"
#include <fstream>

namespace nn
{
    /**
     * @brief Enum with available layer types.
     */
    enum e_layerType { DENSE, BATCH_NORM };

    /**
     * @brief Enum with avaible initializers
     */
    enum e_initializer { HE_NORMAL, HE_UNIFORM, XAVIER_NORMAL, XAVIER_UNIFORM };

    /**
     * @brief Enum with avaible activation functions
     */
    enum e_activation { RELU, SIGMOID, SOFTMAX, NONE };

    /**
     * @class Layer
     * @brief Abstract base class for neural network layers.
     *
     * This class defines the interface for forward and backward propagation,
     * as well as saving layer state
     */
    class Layer
    {
    public:
        /**
         * @brief Performs forward propagation.
         *
         * @param input The input matrix.
         * @return The output matrix after applying the layer's transformation.
         */
        virtual Matrix forward(const Matrix &input) = 0;

        /**
         * @brief Performs backward propagation.
         *
         * @param gradient The gradient of the loss with respect to the output.
         * @return The gradient of the loss with respect to the input.
         */
        virtual Matrix backward(const Matrix &gradient) = 0;

        /**
         * @brief Resets the accumulated gradients of the layer.
         */
        virtual void resetGradients() = 0;

        /**
         * @brief Applies accumulated gradients to the layer.
         * 
         * @param optimizer The optimizer to use for weights and biases updates.
         * @param batchSize Size of the batch from witch gradients were accumulated.
         */
        virtual void applyGradient(Optimizer &optimizer, const int batchSize) = 0;

        /**
         * @brief Saves the layer's state to a binary file.
         *
         * @param file Output file stream (must be opened in binary mode).
         */
        virtual void save(std::ofstream &file) const = 0;

        /**
         * @brief Returns the type of the layer.
         *
         * @return The layer type as an enum value.
         */
        virtual e_layerType getType() const = 0;
    };
}

#endif
