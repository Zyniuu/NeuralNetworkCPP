/**
 * C++ neural network library
 *
 * BinaryCrossEntropy.hpp
 */

#ifndef BINARYCROSSENTROPY_HPP
#define BINARYCROSSENTROPY_HPP

#include "../Common/Loss.hpp"

namespace nn
{
    /**
     * @class BinaryCrossEntropy
     * @brief Implements the Binary Cross-Entropy loss function.
     *
     * Binary Cross-Entropy is used for binary classification tasks.
     * It measures the difference between predicted probabilities and true binary labels.
     */
    class BinaryCrossEntropy : public Loss
    {
    public:
        /**
         * @brief Computes the loss between predictions and targets using BCE formula.
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