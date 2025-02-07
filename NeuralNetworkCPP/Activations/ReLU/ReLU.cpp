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
        return input.map([](double x) { return std::max(0.0, x); });
    }

    Matrix ReLU::backward(const Matrix &gradient)
    {
        // Compute gradient of ReLU: output = (input > 0 ? 1 : 0)
        return gradient.map([](double x) { return (x > 0) ? 1.0 : 0.0; });
    }
}