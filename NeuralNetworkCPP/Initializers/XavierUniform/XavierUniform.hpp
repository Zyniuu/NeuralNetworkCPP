/**
 * C++ neural network library
 *
 * XavierUniform.hpp
 */

#ifndef XAVIERUNIFORM_HPP
#define XAVIERUNIFORM_HPP

#include "../Common/Initializer.hpp"

namespace nn
{
    /**
     * @class XavierUniform
     * @brief Implements Xavier (Glorot) uniform initialization for neural network weights.
     *
     * Xavier Uniform initialization is designed for networks using sigmoid or tanh activation functions.
     * It draws weights from a uniform distribution within the range [-limit, limit],
     * where limit = sqrt(6 / (fan-in + fan-out)), and fan-in and fan-out are the number of input and output neurons.
     */
    class XavierUniform : public Initializer
    {
    private:
        std::uniform_real_distribution<double> m_dist; ///< Uniform distribution for weight initialization

    public:
        /**
         * @brief Constructor for XavierUniform initializer.
         *
         * Initializes the uniform distribution within the range [-limit, limit],
         * where limit = sqrt(6 / (fan-in + fan-out)).
         *
         * @param inputs Number of input neurons (fan-in).
         * @param outputs Number of output neurons (fan-out).
         */
        XavierUniform(const int inputs, const int outputs);

        /**
         * @brief Generates a random number following Xavier uniform distribution.
         *
         * @return A randomly initialized value drawn from the uniform distribution.
         */
        double getRandomNum() override;
    };
}

#endif