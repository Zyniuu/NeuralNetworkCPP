/**
 * C++ neural network library
 *
 * ColumnWiseProxy.cpp
 */

#include "RowWiseProxy.hpp"
#include "../../GlobalThreadPool/GlobalThreadPool.hpp"

namespace nn
{
    RowWiseProxy::RowWiseProxy(Matrix &matrix)
        : m_matrix(matrix) {}
    
    Matrix operator-(const RowWiseProxy &left, Matrix &right)
    {
        // Validate input: `right` must be a row vector with matching columns.
        if (right.getRows() != 1 || right.getCols() != left.m_matrix.getCols())
            throw std::invalid_argument("Row vector dimensions must match matrix columns.");
        
        // Create a copy of the original matrix to store the result.
        Matrix result = left.m_matrix;

        // Get the global thread pool for parallel execution.
        auto &pool = getGlobalThreadPool();

        // Parallelize row-wise subtraction.
        pool.parallelFor(0, result.getRows(), [&result, &right](int i) {
            for (int j = 0; j < result.getCols(); j++)
                result[{i, j}] -= right[{0, j}];
        });

        return result;
    }
}