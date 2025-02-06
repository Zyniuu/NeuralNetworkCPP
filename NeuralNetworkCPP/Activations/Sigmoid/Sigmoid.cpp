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

        return input.map([expLimit](double x) {
            // Clip input values to avoid overflow/underflow in exp
            return 1.0 / (1.0 + std::exp(std::max(-expLimit, std::min(-x, expLimit))));
        });
    }

    Matrix Sigmoid::backward(const Matrix &gradient)
    {
        Matrix output = forward(gradient);

        // Compute gradient of Sigmoid: output = Sigmoid(x) * (1 - Sigmoid(x))
        return output.map([](double x) { 
            return x * (1.0 - x); 
        });
    }
}
