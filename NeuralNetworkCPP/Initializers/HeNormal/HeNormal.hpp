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
     */
    class HeNormal : public Initializer
    {
    private:
        std::normal_distribution<double> m_dist; ///< Normal distribution for weight initialization

    public:
        /**
         * @brief Constructor for HeNormal initializer.
         *
         * @param inputs Number of input neurons.
         * @param outputs Number of output neurons (not used in He Normal).
         */
        HeNormal(const int &inputs, const int &outputs);

        /**
         * @brief Generates a random number following He normal distribution.
         *
         * @return A randomly initialized value.
         */
        double getRandomNum() override;
    };
}

#endif