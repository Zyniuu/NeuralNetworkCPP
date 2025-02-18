/**
 * C++ neural network library
 *
 * ColWiseProxy.cpp
 */

#include "ColWiseProxy.hpp"
#include "../../GlobalThreadPool/GlobalThreadPool.hpp"
#include <limits>

namespace nn
{
    ColWiseProxy::ColWiseProxy(Matrix &matrix)
        : m_matrix(matrix) {}
    
    Matrix ColWiseProxy::maxCoeff() const
    {
        Matrix result(1, m_matrix.getCols(), 0.0);
        auto &pool = getGlobalThreadPool();

        // Iterate over each column
        pool.parallelFor(0, m_matrix.getCols(), [this, &result](int col) {
            double maxVal = std::numeric_limits<double>::lowest();

            // Iterate over each row in the current column
            for (int row = 0; row < m_matrix.getRows(); row++)
            {
                if (m_matrix[{row, col}] > maxVal)
                    maxVal = m_matrix[{row, col}];
            }
            
            // Store the max value in the result row vector
            result[{0, col}] = maxVal;
        });

        return result;
    }

    Matrix ColWiseProxy::sum() const
    {
        Matrix result(1, m_matrix.getCols(), 0.0);
        auto &pool = getGlobalThreadPool();

        // Parallelize the column-wise addition
        pool.parallelFor(0, m_matrix.getCols(), [this, &result](int i) {
            for (int j = 0; j < m_matrix.getRows(); j++)
                result[{0, i}] += m_matrix[{j, i}];
        });

        return result;
    }

    Matrix operator*(const ColWiseProxy &left, Matrix &right)
    {
        // Validate input: `right` must be a column vector with matching rows.
        if (right.getCols() != 1 || right.getRows() != left.m_matrix.getRows())
            throw std::invalid_argument("Column vector dimensions must match matrix rows.");
        
        // Create a copy of the original matrix to store the result.
        Matrix result = left.m_matrix;

        // Get the global thread pool for parallel execution.
        auto &pool = getGlobalThreadPool();

        // Parallelize column-wise multiplication.
        pool.parallelFor(0, result.getCols(), [&result, &right](int i) {
            for (int j = 0; j < result.getRows(); j++)
                result[{j, i}] *= right[{j, 0}];
        });

        return result;
    }

    Matrix operator+(const ColWiseProxy &left, Matrix &right)
    {
        // Validate input: `right` must be a column vector with matching rows.
        if (right.getCols() != 1 || right.getRows() != left.m_matrix.getRows())
            throw std::invalid_argument("Column vector dimensions must match matrix rows.");
        
        // Create a copy of the original matrix to store the result.
        Matrix result = left.m_matrix;

        // Get the global thread pool for parallel execution.
        auto &pool = getGlobalThreadPool();

        // Parallelize column-wise addition.
        pool.parallelFor(0, result.getCols(), [&result, &right](int i) {
            for (int j = 0; j < result.getRows(); j++)
                result[{j, i}] += right[{j, 0}];
        });

        return result;
    }

    Matrix operator-(const ColWiseProxy &left, Matrix &right)
    {
        // Validate input: `right` must be a column vector with matching rows.
        if (right.getCols() != 1 || right.getRows() != left.m_matrix.getRows())
            throw std::invalid_argument("Column vector dimensions must match matrix rows.");
        
        // Create a copy of the original matrix to store the result.
        Matrix result = left.m_matrix;

        // Get the global thread pool for parallel execution.
        auto &pool = getGlobalThreadPool();

        // Parallelize column-wise addition.
        pool.parallelFor(0, result.getCols(), [&result, &right](int i) {
            for (int j = 0; j < result.getRows(); j++)
                result[{j, i}] -= right[{j, 0}];
        });

        return result;
    }
}