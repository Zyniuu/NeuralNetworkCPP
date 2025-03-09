/**
 * C++ neural network library
 *
 * TestUtils.hpp
 */

#ifndef TESTUTILS_HPP
#define TESTUTILS_HPP

#include <vector>
#include <cmath>

/**
 * @brief Computes the standard deviation of a given set of values.
 *
 * @param values Vector of values.
 * @param mean Mean of the values.
 * @return Standard deviation.
 */
inline double calculateStdDev(const std::vector<double> &values, double mean)
{
    double sum = 0.0;
    for (double v : values)
    {
        sum += (v - mean) * (v - mean);
    }
    return std::sqrt(sum / values.size());
}

#endif