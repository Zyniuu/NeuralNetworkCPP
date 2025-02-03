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
        double limit = std::sqrt(6.0 / (inputs + outputs));             // Calculate the limit based on Xavier initialization formula
        m_dist = std::uniform_real_distribution<double>(-limit, limit); // Set the range for uniform distribution
    }

    double XavierUniform::getRandomNum()
    {
        return m_dist(m_gen);
    }
}