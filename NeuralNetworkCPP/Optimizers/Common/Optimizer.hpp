/**
 * C++ neural network library
 *
 * Optimizer.hpp
 */

#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "../../Matrix/Matrix.hpp"

namespace nn
{
    /**
     * @class Optimizer
     * @brief Abstract base class for optimizers.
     */
    class Optimizer
    {
    protected:
        double m_learningRate; ///< Learning rate for parameter updates.

    public:
        /**
         * @brief Constructs an optimizer.
         *
         * @param learningRate The learning rate.
         */
        Optimizer(double learningRate) : m_learningRate(learningRate) {}

        /**
         * @brief Updates the weights and biases of a layer.
         *
         * @param weights The weight matrix to update.
         * @param biases The bias vector to update.
         * @param gradWeights The gradient of the loss with respect to the weights.
         * @param gradBiases The gradient of the loss with respect to the biases.
         */
        virtual void update(Matrix &weights, Matrix &biases, const Matrix &gradWeights, const Matrix &gradBiases) = 0;
    };
}

#endif