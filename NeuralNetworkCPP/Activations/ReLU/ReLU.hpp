/**
 * C++ neural network library
 *
 * ReLU.hpp
 */

#ifndef RELU_HPP
#define RELU_HPP

#include "../Common/Activation.hpp"

namespace nn
{
    /**
     * @class ReLU
     * @brief Implements the Rectified Linear Unit (ReLU) activation function.
     *
     * ReLU is defined as: ReLU(x) = max(0, x)
     */
    class ReLU : public Activation
    {
    public:
        /**
         * @brief Applies the ReLU function to the input matrix.
         *
         * @param input The input matrix.
         * @return The output matrix after applying ReLU.
         */
        Matrix forward(const Matrix &input) override;

        /**
         * @brief Computes the gradient of the ReLU function.
         *
         * @param gradient The gradient of the loss with respect to the output.
         * @return The gradient of the loss with respect to the input.
         */
        Matrix backward(const Matrix &gradient) override;
    };
}

#endif