/**
 * C++ neural network library
 *
 * ColumnWiseProxy.hpp
 */

#ifndef COLUMNWISEPROXY_HPP
#define COLUMNWISEPROXY_HPP

#include "../Matrix.hpp"

namespace nn
{
    // Forward declaration of Matrix class
    class Matrix;

    /**
     * @class ColumnWiseProxy
     * @brief Proxy class for performing column-wise operations on a matrix.
     *
     * This class enables operations like adding a column vector to all columns of a matrix.
     * It is returned by the `Matrix::colwise()` method.
     */
    class ColumnWiseProxy
    {
    private:
        Matrix &m_matrix; ///< Reference to the original matrix.

    public:
        /**
         * @brief Constructs a ColumnWiseProxy for the given matrix.
         *
         * @param matrix The matrix to perform column-wise operations on.
         */
        ColumnWiseProxy(Matrix &matrix);

        /**
         * @brief Adds a column vector to each column of the matrix.
         *
         * @param other A column vector (Matrix with 1 column).
         * @return A new Matrix after the column-wise addition.
         * @throws std::invalid_argument If `other` is not a column vector or its row count
         *                               does not match the original matrix.
         */
        friend Matrix operator+(const ColumnWiseProxy &left, Matrix &right);
    };
}

#endif