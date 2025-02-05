/**
 * C++ neural network library
 *
 * CategoricalCrossEntropy.cpp
 */

#include "CategoricalCrossEntropy.hpp"
#include <cmath>

namespace nn
{
    double CategoricalCrossEntropy::computeLoss(const Matrix &predictions, const Matrix &targets)
    {
        if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols())
            throw std::invalid_argument("Predictions and targets must have the same dimensions.");

        double epsilon = 1e-15;
        double loss = 0.0;
        for (int i = 0; i < targets.getRows(); i++)
        {
            for (int j = 0; j < targets.getCols(); j++)
            {
                if (targets[{i, j}] > 0.0)
                    loss += -(targets[{i, j}] * std::log(predictions[{i, j}] + epsilon)); // Add small number to avoid log(0)
            }
        }

        // Average the loss over all samples
        return loss / targets.getRows();
    }

    Matrix CategoricalCrossEntropy::computeGradient(const Matrix &predictions, const Matrix &targets)
    {
        return (targets * -1) / (predictions + 1e-15); // Add small number to avoid division by zero
    }
}