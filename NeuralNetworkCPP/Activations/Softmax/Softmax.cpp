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
        // Compute gradient of softmax
        Matrix gradInput = gradient;
        for (int i = 0; i < gradient.getRows(); i++)
        {
            for (int j = 0; j < gradient.getCols(); j++)
            {
                double output = m_output[{i, j}];
                gradInput[{i, j}] *= output * (1.0 - output);
            }
        }
        return gradInput;
    }
}