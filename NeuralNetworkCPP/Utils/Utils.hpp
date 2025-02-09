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
}

#endif