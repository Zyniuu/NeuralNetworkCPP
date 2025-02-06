/**
 * C++ neural network library
 *
 * Sigmoid.hpp
 */

#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include "../Common/Activation.hpp"

namespace nn
{
    /**
     * @class Sigmoid
     * @brief Implements the Sigmoid activation function.
     *
     * Sigmoid is defined as: Sigmoid(x) = 1 / (1 + e^{-x})
     */
    class Sigmoid : public Activation
    {
    private:
        Matrix m_output; ///< Stores the output of the forward pass for use in the backward pass.

    public:
        /**
         * @brief Applies the Sigmoid function to the input matrix.
         *
         * @param input The input matrix.
         * @return The output matrix after applying Sigmoid.
         */
        Matrix forward(const Matrix &input) override;

        /**
         * @brief Computes the gradient of the Sigmoid function.
         *
         * @param gradient The gradient of the loss with respect to the output.
         * @return The gradient of the loss with respect to the input.
         */
        Matrix backward(const Matrix &gradient) override;
    };
}

#endif