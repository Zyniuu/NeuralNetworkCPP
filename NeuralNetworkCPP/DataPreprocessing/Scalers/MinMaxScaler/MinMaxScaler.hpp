/**
 * C++ neural network library
 *
 * MinMaxScaler.hpp
 */

#ifndef MINMAXSCALER_HPP
#define MINMAXSCALER_HPP

#include "../Common/Scaler.hpp"

namespace nn
{
    /**
     * @class MinMaxScaler
     * @brief Normalizes data to a specified range (default: [0, 1]).
     */
    class MinMaxScaler : public Scaler
    {
    private:
        std::vector<double> m_min; ///< Minimum value of each feature.
        std::vector<double> m_max; ///< Maximum value of each feature.
        double m_featureRangeMin;  ///< Minimum value of the target range.
        double m_featureRangeMax;  ///< Maximum value of the target range.

    public:
        /**
         * @brief Constructs a MinMaxScaler object.
         *
         * @param featureRangeMin The minimum value of the target range (default: 0.0).
         * @param featureRangeMax The maximum value of the target range (default: 1.0).
         */
        MinMaxScaler(const double featureRangeMin = 0.0, const double featureRangeMax = 1.0);

        /**
         * @brief Fits the scaler to the data (computes min and max).
         *
         * @param data The input data as a vector of vectors of doubles.
         */
        void fit(const std::vector<std::vector<double>> &data) override;

        /**
         * @brief Transforms the data using the computed min and max.
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