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
        double limit = std::sqrt(6.0 / inputs);                         // Compute range for uniform distribution
        m_dist = std::uniform_real_distribution<double>(-limit, limit); // Define uniform distribution
    }

    double HeUniform::getRandomNum()
    {
        return m_dist(m_gen);
    }
}