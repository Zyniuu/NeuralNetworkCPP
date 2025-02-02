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
     */
    class XavierNormal : public Initializer
    {
    private:
        std::normal_distribution<double> m_dist; ///< Normal distribution for weight initialization

    public:
        /**
         * @brief Constructor for XavierNormal initializer.
         *
         * @param inputs Number of input neurons.
         * @param outputs Number of output neurons.
         */
        XavierNormal(const int &inputs, const int &outputs);

        /**
         * @brief Generates a random number following Xavier normal distribution.
         *
         * @return A randomly initialized value.
         */
        double getRandomNum() override;
    };
}

#endif