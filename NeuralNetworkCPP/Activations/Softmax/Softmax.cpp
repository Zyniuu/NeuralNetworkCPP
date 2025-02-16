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
        Matrix expVals = (input - input.maxCoeff()).map([](double x) {
            return std::exp(x);
        });

        m_output = expVals / expVals.sum();

        return m_output;
    }

    Matrix Softmax::backward(const Matrix &gradient)
    {
        // return m_output.map([](double x) { return x * (1 - x); });
        // Create identity matrix
        Matrix id = Matrix::identity(m_output.getRows());
        Matrix outputTansposed = m_output.transpose();
        // Compute the gradient of softmax
        return ((id.rowWise() - outputTansposed).colWise() * m_output) * gradient;
    }
}