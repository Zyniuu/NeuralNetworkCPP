/**
 * C++ neural network library
 *
 * SGD.hpp
 */

#ifndef SGD_HPP
#define SGD_HPP

#include "../Common/Optimizer.hpp"
#include <unordered_map>

namespace nn
{
    /**
     * @class SGD
     * @brief Stochastic Gradient Descent (SGD) optimizer with momentum.
     *
     * This optimizer updates parameters using the gradient of the loss function
     * and a momentum term to accelerate convergence.
     */
    class SGD : public Optimizer
    {
    private:
        double m_momentum; ///< Momentum factor (default: 0.9).

        // Maps to store velocity matrices for weights and biases
        std::unordered_map<Matrix *, Matrix> m_velocityWeights; ///< Velocity for weights.
        std::unordered_map<Matrix *, Matrix> m_velocityBiases;  ///< Velocity for biases.

    public:
        /**
         * @brief Constructs an SGD optimizer.
         *
         * @param learningRate The learning rate (default: 0.001).
         * @param momentum The momentum factor (default: 0.9).
         */
        SGD(double learningRate = 0.001, double momentum = 0.9);

        /**
         * @brief Updates the weights and biases using momentum.
         *
         * @param weights The weight matrix to update.
         * @param biases The bias vector to update.
         * @param gradWeights The gradient of the loss with respect to the weights.
         * @param gradBiases The gradient of the loss with respect to the biases.
         */
        void update(Matrix &weights, Matrix &biases, const Matrix &gradWeights, const Matrix &gradBiases) override;
    };
}

#endif