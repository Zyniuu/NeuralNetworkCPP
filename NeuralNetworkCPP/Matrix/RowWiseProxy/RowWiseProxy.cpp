/**
 * C++ neural network library
 *
 * ColumnWiseProxy.cpp
 */

#include "RowWiseProxy.hpp"
#include "../../GlobalThreadPool/GlobalThreadPool.hpp"

namespace nn
{
    RowWiseProxy::RowWiseProxy(const Matrix &matrix)
        : m_matrix(matrix) {}
    
    Matrix RowWiseProxy::sum() const
    {
        Matrix result(m_matrix.getRows(), 1, 0.0);
        auto &pool = getGlobalThreadPool();

        // Parallelize the row-wise addition
        pool.parallelFor(0, m_matrix.getRows(), [this, &result](int i) {
            for (int j = 0; j < m_matrix.getCols(); j++)
                result[{i, 0}] += m_matrix[{i, j}];
        });

        return result;
    }
    
    Matrix operator-(const RowWiseProxy &left, const Matrix &right)
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

    Matrix operator/(const RowWiseProxy &left, const Matrix &right)
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
                result[{i, j}] /= right[{0, j}];
        });

        return result;
    }
}