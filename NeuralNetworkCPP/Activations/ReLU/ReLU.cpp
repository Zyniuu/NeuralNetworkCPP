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
        Matrix output = input;

        // Apply ReLU element-wise: output = max(0, input)
        return output.map([](double x) { return std::max(0.0, x); });
    }

    Matrix ReLU::backward(const Matrix &gradient)
    {
        Matrix output = gradient;

        // Compute gradient of ReLU: output = (input > 0 ? 1 : 0)
        return output.map([](double x) { return (x > 0) ? 1.0 : 0.0; });
    }
}