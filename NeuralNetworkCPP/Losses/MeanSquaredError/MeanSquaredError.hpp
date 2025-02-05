/**
 * C++ neural network library
 *
 * MeanSquaredError.hpp
 */

#ifndef MEANSQUAREDERROR_HPP
#define MEANSQUAREDERROR_HPP

#include "../Common/Loss.hpp"

namespace nn
{
    /**
     * @class MeanSquaredError
     * @brief Implements the Mean Squared Error (MSE) loss function.
     *
     * MSE is commonly used for regression tasks. It measures the average squared difference
     * between predicted and target values.
     */
    class MeanSquaredError : public Loss
    {
    public:
        /**
         * @brief Computes the loss between predictions and targets using MSE formula.
         *
         * @param predictions The predicted values.
         * @param targets The target values.
         * @return The computed loss.
         */
        double computeLoss(const Matrix &predictions, const Matrix &targets) override;

        /**
         * @brief Computes the gradient of the loss with respect to predictions.
         *
         * @param predictions The predicted values.
         * @param targets The target values.
         * @return The gradient of the loss.
         */
        Matrix computeGradient(const Matrix &predictions, const Matrix &targets) override;
    };
}

#endif