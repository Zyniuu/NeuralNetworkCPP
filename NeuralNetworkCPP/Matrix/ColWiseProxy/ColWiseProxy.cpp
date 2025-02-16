/**
 * C++ neural network library
 *
 * ColWiseProxy.cpp
 */

#include "ColWiseProxy.hpp"
#include "../../GlobalThreadPool/GlobalThreadPool.hpp"

namespace nn
{
    ColWiseProxy::ColWiseProxy(Matrix &matrix)
        : m_matrix(matrix) {}
    
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
}