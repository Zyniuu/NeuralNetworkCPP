/**
 * C++ neural network library
 *
 * HeNormal.hpp
 */

#ifndef HENORMAL_HPP
#define HENORMAL_HPP

#include "../Common/Initializer.hpp"

namespace nn
{
    /**
     * @class HeNormal
     * @brief Implements He Normal initialization for neural network weights.
     *
     * He Normal initialization is designed for networks using ReLU (or variants) activation functions.
     * It draws weights from a normal distribution with a mean of 0 and a standard deviation of sqrt(2 / fan-in),
     * where fan-in is the number of input neurons.
     */
    class HeNormal : public Initializer
    {
    private:
        std::normal_distribution<double> m_dist; ///< Normal distribution for weight initialization

    public:
        /**
         * @brief Constructor for HeNormal initializer.
         *
         * Initializes the normal distribution with a standard deviation of sqrt(2 / fan-in).
         *
         * @param inputs Number of input neurons (fan-in).
         * @param outputs Number of output neurons (fan-out). Not used in He Normal initialization.
         */
        HeNormal(const int inputs, const int outputs);

        /**
         * @brief Generates a random number following He normal distribution.
         *
         * @return A randomly initialized value drawn from the normal distribution.
         */
        double getRandomNum() override;
    };
}

#endif