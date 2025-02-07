/**
 * C++ neural network library
 *
 * Activation.hpp
 */

#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "../../Matrix/Matrix.hpp"

namespace nn
{
    /**
     * @class Activation
     * @brief Abstract base class for activation functions.
     */
    class Activation
    {
    public:
        /**
         * @brief Applies the activation function to the input.
         *
         * @param input The input matrix.
         * @return The output matrix.
         */
        virtual Matrix forward(const Matrix &input) = 0;

        /**
         * @brief Computes the gradient of the activation function.
         *
         * @param gradient The gradient of the loss with respect to the output.
         * @return The gradient of the loss with respect to the input.
         */
        virtual Matrix backward(const Matrix &gradient) = 0;
    };
}

#endif