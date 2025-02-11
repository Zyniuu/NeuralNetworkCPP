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

        // Add small number to avoid log(0)
        Matrix logPred = predictions.map([this](double x) { return std::log(x + m_epsilon); });

        return ((targets.cwiseProduct(logPred)).sum() * -1) / targets.getRows();
    }

    Matrix CategoricalCrossEntropy::computeGradient(const Matrix &predictions, const Matrix &targets)
    {
        if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols())
            throw std::invalid_argument("Predictions and targets must have the same dimensions.");

        return (targets * -1) / (predictions + m_epsilon); // Add small number to avoid division by zero
    }
}