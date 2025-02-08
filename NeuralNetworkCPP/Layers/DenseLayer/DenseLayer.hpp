/**
 * C++ neural network library
 *
 * DenseLayer.hpp
 */

#ifndef DENSELAYER_HPP
#define DENSELAYER_HPP

#include "../Common/Layer.hpp"
#include "../../Activations/Common/Activation.hpp"
#include <memory>

namespace nn
{
    /**
     * @class DenseLayer
     * @brief Implements a fully connected (dense) layer.
     *
     * This layer applies a linear transformation (weights * input + biases) followed by
     * an optional activation function. It supports saving and loading layer state to/from files.
     */
    class DenseLayer : public Layer
    {
    private:
        Matrix m_weights;                         ///< Weight matrix.
        Matrix m_biases;                          ///< Bias vector.
        Matrix m_input;                           ///< Input to the layer (stored for backward pass).
        std::unique_ptr<Activation> m_activation; ///< Optional activation function.
        e_activation m_activationID;              ///< Activation ID used when saving layer to the file

    public:
        /**
         * @brief Constructs a dense layer.
         *
         * @param inputSize Number of input neurons.
         * @param outputSize Number of output neurons.
         * @param initializerID Weights initializer.
         * @param activationID Optional activation function.
         */
        DenseLayer(const int inputSize, const int outputSize, e_initializer initializerID, e_activation activationID);

        /**
         * @brief Constructs a dense layer from the file.
         *
         * @param file Input file stream (must be opened in binary mode).
         * @throws std::runtime_error If the file is not open or reading fails.
         */
        DenseLayer(std::ifstream &file);

        /**
         * @brief Performs forward propagation.
         *
         * @param input The input matrix.
         * @return The output matrix after applying the layer's transformation.
         */
        Matrix forward(const Matrix &input) override;

        /**
         * @brief Performs backward propagation.
         *
         * @param gradient The gradient of the loss with respect to the output.
         * @param optimizer The optimizer to use for weights and biases updates.
         * @return The gradient of the loss with respect to the input.
         */
        Matrix backward(const Matrix &gradient, Optimizer &optimizer) override;

        /**
         * @brief Saves the layer's state to a binary file.
         *
         * @param file Output file stream (must be opened in binary mode).
         * @throws std::runtime_error If the file is not open or reading fails.
         */
        void save(std::ofstream &file) const override;

        /**
         * @brief Returns the type of the layer.
         *
         * @return The layer type as an enum value.
         */
        e_layerType getType() const override { return DENSE; }

    private:
        /**
         * @brief Initializes the weights matrix using the specified initializer.
         *
         * @param inputSize Number of input neurons.
         * @param outputSize Number of output neurons.
         * @param initializerID Weights initializer.
         */
        void initWeights(const int inputSize, const int outputSize, e_initializer initializerID);

        /**
         * @brief Initializes the activation function based on the provided activation ID.
         *
         * @param activationID Activation function ID.
         */
        void initActivationFunction(e_activation activationID);
    };
}

#endif