/**
 * C++ neural network library
 *
 * Scaler.hpp
 */

#ifndef SCALER_HPP
#define SCALER_HPP

#include <vector>

namespace nn
{
    /**
     * @class Scaler
     * @brief Abstract base class for all scalers.
     */
    class Scaler
    {
    public:
        /**
         * @brief Fits the scaler to the data.
         *
         * @param data The input data as a vector of vectors of doubles.
         */
        virtual void fit(const std::vector<std::vector<double>> &data) = 0;

        /**
         * @brief Transforms the data using the fitted parameters.
         *
         * @param data The input data as a vector of vectors of doubles.
         * @return std::vector<std::vector<double>> The normalized data.
         */
        virtual void transform(const std::vector<std::vector<double>> &data) = 0;

        /**
         * @brief Fits the scaler to the data and then transforms the data.
         *
         * @param data The input data as a vector of vectors of doubles.
         * @return std::vector<std::vector<double>> The normalized data.
         */
        virtual void fitTransform(const std::vector<std::vector<double>> &data) = 0;
    };
}

#endif