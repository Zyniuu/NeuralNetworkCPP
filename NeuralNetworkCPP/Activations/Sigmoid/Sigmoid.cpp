/**
 * C++ neural network library
 *
 * Sigmoid.cpp
 */

#include "Sigmoid.hpp"
#include <cmath>

namespace nn
{
    Matrix Sigmoid::forward(const Matrix &input)
    {
        double expLimit = 700; // To avoid overflow/underflow in exp

        m_output = input.map([expLimit](double x) {
            // Clip input values to avoid overflow/underflow in exp
            return 1.0 / (1.0 + std::exp(std::max(-expLimit, std::min(-x, expLimit))));
        });

        return m_output;
    }

    Matrix Sigmoid::backward(const Matrix &gradient)
    {
        // Compute gradient of Sigmoid: gradient * (output * (1 - output))
        return gradient.cwiseProduct(m_output.cwiseProduct(1.0 - m_output));
    }
}
