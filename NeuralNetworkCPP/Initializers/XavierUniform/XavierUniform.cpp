/**
 * C++ neural network library
 *
 * XavierUniform.cpp
 */

#include "XavierUniform.hpp"

namespace nn
{
    XavierUniform::XavierUniform(const int inputs, const int outputs)
        : Initializer(inputs, outputs)
    {
        // Compute the range limit for Xavier Uniform initialization.
        // Formula: limit = sqrt(6.0 / (fan-in + fan-out)), where fan-in and fan-out are input and output neurons.
        double limit = std::sqrt(6.0 / (inputs + outputs));

        // Initialize the uniform distribution within the range [-limit, limit].
        m_dist = std::uniform_real_distribution<double>(-limit, limit);
    }

    double XavierUniform::getRandomNum()
    {
        // Generate a random number from the predefined uniform distribution.
        return m_dist(m_gen);
    }
}