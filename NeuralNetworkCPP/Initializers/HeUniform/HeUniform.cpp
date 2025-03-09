/**
 * C++ neural network library
 *
 * HeUniform.cpp
 */

#include "HeUniform.hpp"

namespace nn
{
    HeUniform::HeUniform(const int inputs, const int outputs)
        : Initializer(inputs, outputs)
    {
        // Compute the range limit for He Uniform initialization.
        // Formula: limit = sqrt(6.0 / fan-in), where fan-in is the number of input neurons.
        double limit = std::sqrt(6.0 / inputs);

        // Initialize the uniform distribution within the range [-limit, limit]
        m_dist = std::uniform_real_distribution<double>(-limit, limit);
    }

    double HeUniform::getRandomNum()
    {
        // Generate a random number from the predefined uniform distribution.
        return m_dist(m_gen);
    }
}