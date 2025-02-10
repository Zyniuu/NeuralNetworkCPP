/**
 * C++ neural network library
 *
 * MeanSquaredError.cpp
 */

#include "MeanSquaredError.hpp"
#include <numeric>

namespace nn
{
    double MeanSquaredError::computeLoss(const Matrix &predictions, const Matrix &targets)
    {
        Matrix error = (targets - predictions).map([](double x) { 
            return x * x; 
        });

        return error.sum() / (error.getRows() * error.getCols());
    }

    Matrix MeanSquaredError::computeGradient(const Matrix &predictions, const Matrix &targets)
    {
        double scale = 2.0 / (predictions.getRows() * predictions.getCols());
        return scale * (predictions - targets);
    }
}