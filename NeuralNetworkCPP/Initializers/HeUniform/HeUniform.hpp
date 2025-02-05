/**
 * C++ neural network library
 *
 * HeUniform.hpp
 */

#ifndef HEUNIFORM_HPP
#define HEUNIFORM_HPP

#include "../Common/Initializer.hpp"

namespace nn
{
    /**
     * @class HeUniform
     * @brief Implements He Uniform initialization for neural network weights.
     *
     * He Uniform initialization is designed for networks using ReLU (or variants) activation functions.
     * It draws weights from a uniform distribution within the range [-limit, limit],
     * where limit = sqrt(6 / fan-in), and fan-in is the number of input neurons.
     */
    class HeUniform : public Initializer
    {
    private:
        std::uniform_real_distribution<double> m_dist; ///< Uniform distribution for weight initialization

    public:
        /**
         * @brief Constructor for HeUniform initializer.
         *
         * Initializes the uniform distribution within the range [-limit, limit],
         * where limit = sqrt(6 / fan-in).
         *
         * @param inputs Number of input neurons (fan-in).
         * @param outputs Number of output neurons (fan-out). Not used in He Uniform initialization.
         */
        HeUniform(const int inputs, const int outputs);

        /**
         * @brief Generates a random number following He uniform distribution.
         *
         * @return A randomly initialized value drawn from the uniform distribution.
         */
        double getRandomNum() override;
    };
}

#endif