/**
 * C++ neural network library
 *
 * Initializer.hpp
 */

#ifndef INITIALIZER_HPP
#define INITIALIZER_HPP

#include <random>

namespace nn
{
    /**
     * @class Initializer
     * @brief Abstract base class for weight initializers in neural network.
     */
    class Initializer
    {
    protected:
        int m_inputs;       ///< Number of input neurons
        int m_outputs;      ///< Number of output neurons
        std::mt19937 m_gen; ///< Random number generator

    public:
        /**
         * @brief Constructor for the Initializer class.
         *
         * @param inputs Number of input neurons.
         * @param outputs Number of output neurons.
         */
        Initializer(const int &inputs, const int &outputs)
            : m_inputs(inputs), m_outputs(outputs), m_gen(std::random_device{}()) {}

        /**
         * @brief Pure virtual function for generating a random number.
         *
         * @return A randomly initialized value.
         */
        virtual double getRandomNum() = 0;
    };
}

#endif