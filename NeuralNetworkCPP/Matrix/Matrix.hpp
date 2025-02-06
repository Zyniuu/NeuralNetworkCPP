/**
 * C++ neural network library
 *
 * Matrix.hpp
 */

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <functional>
#include <iostream>
#include <fstream>
#include "ColumnWiseProxy/ColumnWiseProxy.hpp"

namespace nn
{
    // Forward declaration of ColumnWiseProxy
    class ColumnWiseProxy;

    /**
     * @class Matrix
     * @brief Represents a mathematical matrix with element-wise operations.
     *
     * This class provides functionality for matrix operations such as addition,
     * subtraction, multiplication, division, and element-wise operations.
     * It also supports parallel execution using a global thread pool.
     */
    class Matrix
    {
    private:
        int m_rows;                 ///< Number of rows in the matrix.
        int m_cols;                 ///< Number of columns in the matrix
        std::vector<double> m_data; ///< Matrix data stored in a 1D vector

    public:
        /** @brief Default constructor, creates an empty matrix. */
        Matrix();

        /** @brief Copy constructor. */
        Matrix(const Matrix &matrix);

        /** @brief Move constructor. */
        Matrix(Matrix &&matrix);

        /**
         * @brief Constructs a matrix with given dimensions and an initial value.
         *
         * @param rows Number of rows.
         * @param cols Number of columns.
         * @param initVal Initial value for all elements (default: 0).
         */
        Matrix(const int rows, const int cols, const double initVal = 0.0);

        /**
         * @brief Constructs a matrix from a vector of values.
         *
         * @param rows Number of rows.
         * @param cols Number of columns.
         * @param data Vector containing matrix elements.
         * @throws std::invalid_argument if the data size does not match dimensions.
         */
        Matrix(const int rows, const int cols, const std::vector<double> &data);

        /**
         * @brief Constructs a matrix with values generated by a function.
         *
         * @param rows Number of rows.
         * @param cols Number of columns.
         * @param func Function that generates values for each element.
         */
        Matrix(const int rows, const int cols, std::function<double()> func);

        /**
         * @brief Constructs a matrix from a binary file.
         * 
         * @param file Input file stream (must be opened in binary mode).
         * @throws std::runtime_error If the file is not open or reading fails.
         */
        Matrix(std::ifstream &file);

        /** @brief Destructor (default). */
        ~Matrix() = default;

        /** @brief Returns the number of rows. */
        int getRows() const { return m_rows; }

        /** @brief Returns the number of columns. */
        int getCols() const { return m_cols; }

        /** @brief Returns the matrix data as a vector. */
        std::vector<double> getData() const { return m_data; }

        /**
         * @brief Saves the matrix to a binary file.
         *
         * @param file Output file stream (must be opened in binary mode).
         * @throws std::runtime_error If the file is not open or writing fails.
         */
        void save(std::ofstream &file) const;

        /**
         * @brief Returns a ColumnWiseProxy to enable column-wise operations.
         *
         * @return A ColumnWiseProxy object bound to this matrix.
         */
        ColumnWiseProxy colwise();

        /** @brief Performs element-wise multiplication (Hadamard product). */
        Matrix cwiseProduct(const Matrix &other) const;

        /** @brief Returns the transposed matrix. */
        Matrix transpose();

        /** @brief Applies a function to each matrix element. */
        Matrix map(std::function<double(double)> func);

        Matrix &operator=(const Matrix &other);

        /** @brief Accesses elements using [{row, column}] pair notation. */
        double &operator[](const std::pair<int, int> &index);
        const double &operator[](const std::pair<int, int> &index) const;

        /** @brief Accesses elements using (row, column) notation. */
        double &operator()(const int row, const int col);
        const double &operator()(const int row, const int col) const;

        /** @brief Matrix-matrix and matrix-scalar operations. */
        Matrix &operator+=(const Matrix &other);
        Matrix &operator+=(const double scalar);
        Matrix &operator-=(const Matrix &other);
        Matrix &operator-=(const double scalar);
        Matrix &operator*=(const Matrix &other);
        Matrix &operator*=(const double scalar);
        Matrix &operator/=(const Matrix &other);
        Matrix &operator/=(const double scalar);

        /** @brief Prints the matrix to the output stream. */
        friend std::ostream &operator<<(std::ostream &out, const Matrix &m);

        /** @brief Arithmetic operations (non-modifying). */
        friend Matrix operator*(const Matrix &left, const Matrix &right);
        friend Matrix operator*(const double scalar, const Matrix &right);
        friend Matrix operator*(const Matrix &left, const double scalar);
        friend Matrix operator+(const Matrix &left, const Matrix &right);
        friend Matrix operator+(const double scalar, const Matrix &right);
        friend Matrix operator+(const Matrix &left, const double scalar);
        friend Matrix operator-(const Matrix &left, const Matrix &right);
        friend Matrix operator-(const Matrix &left, const double scalar);
        friend Matrix operator-(const double scalar, const Matrix &right);
        friend Matrix operator/(const Matrix &left, const Matrix &right);
        friend Matrix operator/(const Matrix &left, const double scalar);

        /** @brief Comparison operators. */
        friend bool operator==(const Matrix &left, const Matrix &right);
        friend bool operator!=(const Matrix &left, const Matrix &right);
    };
}

#endif