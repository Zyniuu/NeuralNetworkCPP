/**
 * C++ neural network library
 *
 * RowWiseProxy.hpp
 */

#ifndef ROWWISEPROXY_HPP
#define ROWWISEPROXY_HPP

#include "../Matrix.hpp"

namespace nn
{
    // Forward declaration of Matrix class
    class Matrix;

    /**
     * @class RowWiseProxy
     * @brief Proxy class for performing row-wise operations on a matrix.
     *
     * This class enables operations like adding a row vector to all rows of a matrix.
     * It is returned by the `Matrix::rowWise()` method.
     */
    class RowWiseProxy
    {
    private:
        Matrix &m_matrix; ///< Reference to the original matrix.

    public:
        /**
         * @brief Constructs a RowWiseProxy for the given matrix.
         *
         * @param matrix The matrix to perform row-wise operations on.
         */
        RowWiseProxy(Matrix &matrix);

        /**
         * @brief Adds a row vector to each row of the matrix.
         *
         * @param other A row vector (Matrix with 1 row).
         * @return A new Matrix after the row-wise addition.
         * @throws std::invalid_argument If `other` is not a row vector or its column count
         *                               does not match the original matrix.
         */
        friend Matrix operator+(const RowWiseProxy &left, Matrix &right);
    };
}

#endif