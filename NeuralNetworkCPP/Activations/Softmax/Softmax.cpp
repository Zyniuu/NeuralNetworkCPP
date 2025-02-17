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
        Matrix tmp = input;
        RowWiseProxy rowWiseInput = tmp.rowWise();
        Matrix maxCoeffs = tmp.colWise().maxCoeff();

        // Shift input values by subtracting the maximum value to avoid overflow in exp
        // and then apply softmax
        Matrix expVals = (rowWiseInput - maxCoeffs).map([](double x) {
            return std::exp(x);
        });

        RowWiseProxy rowWiseExpVals = expVals.rowWise();
        Matrix colWiseSum = expVals.colWise().sum();
        m_output = rowWiseExpVals / colWiseSum;

        return m_output;
    }

    Matrix Softmax::backward(const Matrix &gradient)
    {
        // Compute the gradient of softmax
        return m_output.map([](double x) { return x * (1 - x); });
    }
}