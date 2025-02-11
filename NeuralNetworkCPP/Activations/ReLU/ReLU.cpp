/**
 * C++ neural network library
 *
 * ReLU.cpp
 */

#include "ReLU.hpp"

namespace nn
{
    Matrix ReLU::forward(const Matrix &input)
    {
        // Apply ReLU element-wise: output = max(0, input)
        m_output = input.map([](double x) { return std::max(0.0, x); });
        return m_output;
    }

    Matrix ReLU::backward(const Matrix &gradient)
    {
        return gradient.cwiseProduct(
            // Compute gradient of ReLU: output = (input > 0 ? 1 : 0)
            m_output.map([](double x) { return (x > 0) ? 1.0 : 0.0; })
        );
    }
}