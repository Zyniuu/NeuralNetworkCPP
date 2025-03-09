/**
 * C++ neural network library
 *
 * XavierNormal.cpp
 */

#include "XavierNormal.hpp"

namespace nn
{
    XavierNormal::XavierNormal(const int inputs, const int outputs)
        : Initializer(inputs, outputs)
    {
        // Compute the standard deviation for Xavier Normal initialization.
        // Formula: sigma = sqrt(2.0 / (fan-in + fan-out)), where fan-in and fan-out are input and output neurons.
        double sigma = std::sqrt(2.0 / (m_inputs + m_outputs));

        // Initialize the normal distribution with mean 0 and computed standard deviation.
        m_dist = std::normal_distribution<double>(0, sigma);
    }

    double XavierNormal::getRandomNum()
    {
        // Generate a random number from the predefined normal distribution.
        return m_dist(m_gen);
    }
}