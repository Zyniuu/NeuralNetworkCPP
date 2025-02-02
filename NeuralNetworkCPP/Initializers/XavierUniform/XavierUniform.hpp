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
     */
    class XavierUniform : public Initializer
    {
    private:
        std::uniform_real_distribution<double> m_dist; ///< Uniform distribution for weight initialization

    public:
        /**
         * @brief Constructor for XavierUniform initializer.
         *
         * @param inputs Number of input neurons.
         * @param outputs Number of output neurons.
         */
        XavierUniform(const int &inputs, const int &outputs);

        /**
         * @brief Generates a random number following Xavier uniform distribution.
         *
         * @return A randomly initialized value.
         */
        double getRandomNum() override;
    };
}

#endif