/**
 * C++ neural network library
 *
 * Adam.hpp
 */

#ifndef ADAM_HPP
#define ADAM_HPP

#include "../Common/Optimizer.hpp"
#include <unordered_map>

namespace nn
{
    /**
     * @class Adam
     * @brief Adam optimizer.
     *
     * This optimizer combines the benefits of momentum and RMSprop to achieve
     * faster convergence and better performance on a wide range of problems.
     */
    class Adam : public Optimizer
    {
    private:
        double m_beta1;   ///< Exponential decay rate for the first moment estimates.
        double m_beta2;   ///< Exponential decay rate for the second moment estimates.
        double m_epsilon; ///< Small constant for numerical stability.
        int m_t;          ///< Time step (for bias correction).

        // Maps for first and second moment estimates
        std::unordered_map<Matrix *, Matrix> m_m; ///< First moment estimates for weights and biases
        std::unordered_map<Matrix *, Matrix> m_v; ///< Second moment estimates for weights and biases

    public:
        /**
         * @brief Constructs an Adam optimizer.
         *
         * @param learningRate The learning rate (default: 0.001).
         * @param beta1 Exponential decay rate for the first moment estimates (default: 0.9).
         * @param beta2 Exponential decay rate for the second moment estimates (default: 0.999).
         * @param epsilon Small constant for numerical stability (default: 1e-8).
         */
        Adam(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);

        /**
         * @brief Updates the weights and biases using Adam.
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