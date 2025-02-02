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
     */
    class HeUniform : public Initializer
    {
    private:
        std::uniform_real_distribution<double> m_dist; ///< Uniform distribution for weight initialization

    public:
        /**
         * @brief Constructor for HeUniform initializer.
         *
         * @param inputs Number of input neurons.
         * @param outputs Number of output neurons (not used in He Uniform).
         */
        HeUniform(const int &inputs, const int &outputs);

        /**
         * @brief Generates a random number following He uniform distribution.
         *
         * @return A randomly initialized value.
         */
        double getRandomNum() override;
    };
}

#endif