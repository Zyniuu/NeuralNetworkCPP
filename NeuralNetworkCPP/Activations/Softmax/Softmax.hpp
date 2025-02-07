/**
 * C++ neural network library
 *
 * Softmax.hpp
 */

#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "../Common/Activation.hpp"

namespace nn
{
    /**
     * @class Softmax
     * @brief Implements the Softmax activation function.
     *
     * Softmax is defined as: exp(x_i) / {sum_{j} exp(x_j)}
     */
    class Softmax : public Activation
    {
    private:
        Matrix m_output; ///< Stores the output of the forward pass for use in the backward pas

    public:
        /**
         * @brief Applies the Softmax function to the input matrix.
         *
         * @param input The input matrix.
         * @return The output matrix after applying Softmax.
         */
        Matrix forward(const Matrix &input) override;

        /**
         * @brief Computes the gradient of the Softmax function.
         *
         * @param gradient The gradient of the loss with respect to the output.
         * @return The gradient of the loss with respect to the input.
         */
        Matrix backward(const Matrix &gradient) override;
    };
}

#endif