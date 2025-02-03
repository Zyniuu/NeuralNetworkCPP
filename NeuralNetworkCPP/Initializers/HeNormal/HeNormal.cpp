/**
 * C++ neural network library
 *
 * HeNormal.cpp
 */

#include "HeNormal.hpp"

namespace nn
{
    HeNormal::HeNormal(const int inputs, const int outputs)
        : Initializer(inputs, outputs)
    {
        double sigma = std::sqrt(2.0 / m_inputs);            // Compute standard deviation for He initialization
        m_dist = std::normal_distribution<double>(0, sigma); // Define the normal distribution
    }

    double HeNormal::getRandomNum()
    {
        return m_dist(m_gen);
    }
}