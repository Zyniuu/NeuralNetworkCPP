/**
 * C++ neural network library
 *
 * Loss.hpp
 */

#ifndef LOSS_HPP
#define LOSS_HPP

#include "../../Matrix/Matrix.hpp"

namespace nn
{
    /**
     * @class Loss
     * @brief Abstract base class for loss functions.
     *
     * This class defines the interface for computing loss and its gradient.
     */
    class Loss
    {
    protected:
        double m_epsilon = 1e-15; ///< Small number for numerical stability

    public:
        /**
         * @brief Computes the loss between predictions and targets.
         *
         * @param predictions The predicted values.
         * @param targets The target values.
         * @return The computed loss.
         */
        virtual double computeLoss(const Matrix &predictions, const Matrix &targets) = 0;

        /**
         * @brief Computes the gradient of the loss with respect to predictions.
         *
         * @param predictions The predicted values.
         * @param targets The target values.
         * @return The gradient of the loss.
         */
        virtual Matrix computeGradient(const Matrix &predictions, const Matrix &targets) = 0;
    };
}

#endif