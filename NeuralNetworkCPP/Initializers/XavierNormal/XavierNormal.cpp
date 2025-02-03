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
        double sigma = std::sqrt(2.0 / (m_inputs + m_outputs));
        m_dist = std::normal_distribution<double>(0, sigma);
    }

    double XavierNormal::getRandomNum()
    {
        return m_dist(m_gen); // Generates a random number from the predefined normal distribution
    }
}