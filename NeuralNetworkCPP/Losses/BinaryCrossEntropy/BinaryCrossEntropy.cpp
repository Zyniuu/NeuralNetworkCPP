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
        
        double epsilon = 1e-15;
        double loss = 0.0;
        for (int i = 0; i < targets.getRows(); i++)
        {
            for (int j = 0; j < targets.getCols(); j++)
            {
                double y = targets[{i, j}];
                double p = predictions[{i, j}];
                loss += -(y * std::log(p + epsilon) + (1 - y) * std::log(1 - p + epsilon));
            }
        }

        // Average the loss over all samples
        return loss / predictions.getRows();
    }

    Matrix BinaryCrossEntropy::computeGradient(const Matrix &predictions, const Matrix &targets)
    {
        return ((targets / (predictions + 1e-15)) - ((1 - targets) / (1 - predictions + 1e-15))) * -1;
    }
}