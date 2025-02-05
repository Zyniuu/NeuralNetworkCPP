/**
 * C++ neural network library
 *
 * CategoricalCrossEntropy.hpp
 */

#ifndef CATEGORICALCROSSENTROPY_HPP
#define CATEGORICALCROSSENTROPY_HPP

#include "../Common/Loss.hpp"

namespace nn
{
    /**
     * @class CategoricalCrossEntropy
     * @brief Implements the Categorical Cross-Entropy loss function.
     *
     * Categorical Cross-Entropy is used for multi-class classification tasks.
     * It measures the difference between predicted class probabilities and true class labels.
     */
    class CategoricalCrossEntropy : public Loss
    {
    public:
        /**
         * @brief Computes the loss between predictions and targets using CCE formula.
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