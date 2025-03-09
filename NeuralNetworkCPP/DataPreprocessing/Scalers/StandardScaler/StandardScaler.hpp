/**
 * C++ neural network library
 *
 * StandardScaler.hpp
 */

#ifndef STANDARDSCALER_HPP
#define STANDARDSCALER_HPP

#include "../Common/Scaler.hpp"

namespace nn
{
    /**
     * @class StandardScaler
     * @brief Normalizes data to have a mean of 0 and a standard deviation of 1.
     */
    class StandardScaler : public Scaler
    {
    private:
        std::vector<double> m_mean;   ///< Mean of each feature.
        std::vector<double> m_stddev; ///< Standard deviation of each feature.

    public:
        /**
         * @brief Fits the scaler to the data (computes mean and standard deviation).
         *
         * @param data The input data as a vector of vectors of doubles.
         */
        void fit(const std::vector<std::vector<double>> &data) override;

        /**
         * @brief Transforms the data using the computed mean and standard deviation.
         *
         * @param data The input data as a vector of vectors of doubles.
         * @return std::vector<std::vector<double>> The normalized data.
         */
        std::vector<std::vector<double>> transform(const std::vector<std::vector<double>> &data) override;

        /**
         * @brief Fits the scaler to the data and then transforms the data.
         *
         * @param data The input data as a vector of vectors of doubles.
         * @return std::vector<std::vector<double>> The normalized data.
         */
        std::vector<std::vector<double>> fitTransform(const std::vector<std::vector<double>> &data) override;
    };
}

#endif