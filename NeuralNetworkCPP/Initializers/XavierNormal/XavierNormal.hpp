/**
 * C++ neural network library
 *
 * XavierNormal.hpp
 */

#ifndef XAVIERNORMAL_HPP
#define XAVIERNORMAL_HPP

#include "../Common/Initializer.hpp"

namespace nn
{
    /**
     * @class XavierNormal
     * @brief Implements Xavier (Glorot) normal initialization for neural network weights.
     *
     * Xavier Normal initialization is designed for networks using sigmoid or tanh activation functions.
     * It draws weights from a normal distribution with a mean of 0 and a standard deviation of sqrt(2 / (fan-in + fan-out)),
     * where fan-in is the number of input neurons and fan-out is the number of output neurons.
     */
    class XavierNormal : public Initializer
    {
    private:
        std::normal_distribution<double> m_dist; ///< Normal distribution for weight initialization

    public:
        /**
         * @brief Constructor for XavierNormal initializer.
         *
         * Initializes the normal distribution with a standard deviation of sqrt(2 / (fan-in + fan-out)).
         *
         * @param inputs Number of input neurons (fan-in).
         * @param outputs Number of output neurons (fan-out).
         */
        XavierNormal(const int inputs, const int outputs);

        /**
         * @brief Generates a random number following Xavier normal distribution.
         *
         * @return A randomly initialized value drawn from the normal distribution.
         */
        double getRandomNum() override;
    };
}

#endif