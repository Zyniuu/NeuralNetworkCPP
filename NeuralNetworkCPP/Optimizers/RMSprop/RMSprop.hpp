/**
 * C++ neural network library
 *
 * RMSprop.hpp
 */

#ifndef RMSPROP_HPP
#define RMSPROP_HPP

#include "../Common/Optimizer.hpp"
#include <unordered_map>

namespace nn
{
    /**
     * @class RMSprop
     * @brief RMSprop optimizer.
     *
     * This optimizer divides the learning rate by an exponentially decaying average
     * of squared gradients to normalize the updates.
     */
    class RMSprop : public Optimizer
    {
    private:
        double m_gamma;   ///< Decay rate for the moving average of squared gradients.
        double m_epsilon; ///< Small constant for numerical stability.

        // Maps to store moving averages of squared gradients
        std::unordered_map<Matrix *, Matrix> m_vWeights; ///< Moving average for weights.
        std::unordered_map<Matrix *, Matrix> m_vBiases;  ///< Moving average for biases.

    public:
        /**
         * @brief Constructs an RMSprop optimizer.
         *
         * @param learningRate The learning rate (default: 0.001).
         * @param gamma Decay rate for the moving average (default: 0.9).
         * @param epsilon Small constant for numerical stability (default: 1e-8).
         */
        RMSprop(double learningRate = 0.001, double gamma = 0.9, double epsilon = 1e-8);

        /**
         * @brief Updates the weights and biases using RMSprop.
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