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
        // Compute the standard deviation for He Normal initialization.
        // Formula: sigma = sqrt(2.0 / fan-in), where fan-in is the number of input neurons.
        double sigma = std::sqrt(2.0 / m_inputs);

        // Initialize the normal distribution with mean 0 and computed standard deviation.
        m_dist = std::normal_distribution<double>(0, sigma);
    }

    double HeNormal::getRandomNum()
    {
        // Generate a random number from the predefined normal distribution.
        return m_dist(m_gen);
    }
}