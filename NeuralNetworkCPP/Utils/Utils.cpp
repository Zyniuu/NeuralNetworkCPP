/**
 * C++ neural network library
 *
 * Utils.cpp
 */

#include "Utils.hpp"
#include <stdexcept>
#include <numeric>
#include <random>
#include <algorithm>

namespace nn
{
    std::vector<std::vector<double>> slice(const std::vector<std::vector<double>> &data, const int start, const int end)
    {
        // Check if indices are within bounds
        if (start < 0 || end > data.size() || start > end)
            throw std::out_of_range("Invalid start or end index for slicing.");

        // Create a new vector containing the sliced data
        std::vector<std::vector<double>> result(data.begin() + start, data.begin() + end);
        return result;
    }

    void reorderRows(std::vector<std::vector<double>> &data, const std::vector<int> &order)
    {
        // Check if the order vector has the same size as the data
        if (order.size() != data.size())
            throw std::out_of_range("Order vector size does not match data size.");
        
        // Create a copy of the original data
        std::vector<std::vector<double>> originalData = data;

        // Reorder the rows of the data vector
        for (int i = 0; i < order.size(); i++)
        {
            // Ensure the index is within bounds
            if (order[i] < 0 || order[i] >= data.size())
                throw std::out_of_range("Invalid index in order vector.");

            // Place the row in the correct position
            data[i] = originalData[order[i]];
        }
    }

    void shuffleDataset(std::vector<std::vector<double>> &data, std::vector<std::vector<double>> &labels)
    {
        // Check if data and labels have the same number of rows
        if (data.size() != labels.size())
            throw std::runtime_error("Data vector and labels vector must have the same amount of rows.");
        
        // Create vector of shuffled indices
        std::vector<int> indices(data.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        // Shuffle the indices vector
        std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device{}()));

        // Reorder the rows of data and labels
        reorderRows(data, indices);
        reorderRows(labels, indices);
    }
}