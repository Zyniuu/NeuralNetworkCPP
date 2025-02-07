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
        Matrix gradInput = gradient;

        // Compute gradient of Sigmoid: gradInput = gradient * (output * (1 - output))
        for (int i = 0; i < gradInput.getRows(); i++)
        {
            for (int j = 0; j < gradInput.getCols(); j++)
            {
                double output = m_output[{i, j}];
                gradInput[{i, j}] *= output * (1.0 - output);
            }
        }

        return gradInput;
    }
}
