/**
 * C++ neural network library
 *
 * Utils.hpp
 */

#ifndef UTILS_HPP
#define UTILS_HPP

/**
 * @file Utils.hpp
 * @brief This file contains helper functions.
 */

#include <vector>

namespace nn
{
    /**
     * @brief Slices a 2D vector from index `start` to `end` (exclusive).
     *
     * @param data The input 2D vector to slice.
     * @param start The starting index (inclusive).
     * @param end The ending index (exclusive).
     * @return A new 2D vector containing the sliced data.
     * @throws std::out_of_range If `start` or `end` are out of bounds.
     */
    std::vector<std::vector<double>> slice(const std::vector<std::vector<double>> &data, const int start, const int end);

    /**
     * @brief Reorders the rows of a 2D vector based on the provided order.
     *
     * @param data The input 2D vector to reorder.
     * @param order A vector of indices specifying the new order of rows.
     * @throws std::out_of_range If any index in `order` is out of bounds.
     */
    void reorderRows(std::vector<std::vector<double>> &data, const std::vector<int> &order);

    /**
     * @brief Shuffles the rows of the provided data and labels.
     *
     * @param data A 2D data vector to shuffle.
     * @param labels A 2D labels vector to shuffle.
     * @throws std::runtime_error If data and labels have different amount of rows.
     */
    void shuffleDataset(std::vector<std::vector<double>> &data, std::vector<std::vector<double>> &labels);

    /**
     * @brief Converts a vector of class labels into one-hot encoded vectors.
     *
     * @param data The input data as a vector of vectors of doubles (class labels).
     * @param numClasses The number of classes. If 0, it is determined automatically.
     * @return std::vector<std::vector<double>> The one-hot encoded data.
     */
    std::vector<std::vector<double>> to_categorical(const std::vector<std::vector<double>> &data, int numClasses = 0);
}

#endif