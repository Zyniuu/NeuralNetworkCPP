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
     * @brief Abstract base class for weight initializers in neural networks.
     *
     * This class provides a common interface for weight initialization strategies.
     * Derived classes implement specific initialization methods (e.g., He Normal, Xavier Uniform).
     */
    class Initializer
    {
    protected:
        int m_inputs;       ///< Number of input neurons
        int m_outputs;      ///< Number of output neurons
        std::mt19937 m_gen; ///< Mersenne Twister random number generator.

    public:
        /**
         * @brief Constructor for the Initializer class.
         *
         * Initializes the random number generator and stores the number of input and output neurons.
         *
         * @param inputs Number of input neurons.
         * @param outputs Number of output neurons.
         */
        Initializer(const int inputs, const int outputs)
            : m_inputs(inputs), m_outputs(outputs), m_gen(std::random_device{}()) {}

        /**
         * @brief Pure virtual function for generating a random number.
         *
         * Derived classes must implement this method to provide specific initialization logic.
         *
         * @return A randomly initialized value.
         */
        virtual double getRandomNum() = 0;
    };
}

#endif