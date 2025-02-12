/**
 * C++ neural network library
 *
 * BinaryCrossEntropy.cpp
 */

#include "BinaryCrossEntropy.hpp"
#include <cmath>

namespace nn
{
    double BinaryCrossEntropy::computeLoss(const Matrix &predictions, const Matrix &targets)
    {
        if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols())
            throw std::invalid_argument("Predictions and targets must have the same dimensions.");
        
        Matrix logPred = predictions.map([this](double x) { return std::log(x + m_epsilon); });
        Matrix logOneMinusPred = (1 - predictions).map([this](double x) { return std::log(x + m_epsilon); });

        return ((targets.cwiseProduct(logPred) + (1 - targets).cwiseProduct(logOneMinusPred)).sum() * -1) / targets.getRows();
    }

    Matrix BinaryCrossEntropy::computeGradient(const Matrix &predictions, const Matrix &targets)
    {
        if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols())
            throw std::invalid_argument("Predictions and targets must have the same dimensions.");

        return (-1 * (targets / (predictions + m_epsilon))) + ((1 - targets) / (1 - predictions + m_epsilon));
    }
}