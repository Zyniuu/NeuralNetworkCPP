/**
 * C++ neural network library
 *
 * Softmax.cpp
 */

#include "Softmax.hpp"
#include <cmath>

namespace nn
{
    Matrix Softmax::forward(const Matrix &input)
    {
        // Shift input values by subtracting the maximum value to avoid overflow in exp
        // and then apply softmax
        Matrix expVals = (input.rowWise() - input.colWise().maxCoeff()).map([](double x) {
            return std::exp(x);
        });

        m_output = expVals.rowWise() / expVals.colWise().sum();

        return m_output;
    }

    Matrix Softmax::backward(const Matrix &gradient)
    {
        // Compute the gradient of softmax
        return m_output.map([](double x) { return x * (1 - x); });
    }
}