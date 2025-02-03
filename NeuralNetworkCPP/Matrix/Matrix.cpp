/**
 * C++ neural network library
 *
 * Matrix.cpp
 */

#include "Matrix.hpp"
#include "../ThreadPool/ThreadPool.hpp"
#include <iomanip>

namespace nn
{
    Matrix::Matrix()
        : m_rows(0), m_cols(0), m_data({}) {}

    Matrix::Matrix(const Matrix &matrix)
        : m_rows(matrix.m_rows), m_cols(matrix.m_cols), m_data(matrix.m_data) {}

    Matrix::Matrix(Matrix &&matrix)
        : m_rows(matrix.m_rows), m_cols(matrix.m_cols), m_data(std::move(matrix.m_data)) {}

    Matrix::Matrix(const int rows, const int cols, double initVal)
        : m_rows(rows), m_cols(cols), m_data(rows * cols, initVal) {}

    Matrix::Matrix(const int rows, const int cols, const std::vector<double> &data)
        : m_rows(rows), m_cols(cols), m_data(data)
    {
        if (data.size() != rows * cols)
            throw std::invalid_argument("Data size does not match matrix dimensions.");
    }

    Matrix::Matrix(const int rows, const int cols, std::function<double()> func)
        : m_rows(rows), m_cols(cols), m_data(rows * cols)
    {
        ThreadPool::parallelFor(0, rows * cols, [&](int i) { 
            m_data[i] = func();
        });
    }

    double &Matrix::operator[](const std::pair<int, int> &index)
    {
        return m_data[index.first * m_cols + index.second];
    }

    const double &Matrix::operator[](const std::pair<int, int> &index) const
    {
        return m_data[index.first * m_cols + index.second];
    }

    double &Matrix::operator()(const int row, const int col)
    {
        return m_data[row * m_cols + col];
    }

    const double &Matrix::operator()(const int row, const int col) const
    {
        return m_data[row * m_cols + col];
    }

    Matrix Matrix::cwiseProduct(const Matrix &other)
    {
        if (m_rows != other.m_rows || m_cols != other.m_cols)
            throw std::invalid_argument("Matrix dimensions must match for element-wise multiplication.");

        Matrix result(m_rows, m_cols, 0.0);

        ThreadPool::parallelFor(0, m_rows * m_cols, [&](int i) { 
            result.m_data[i] = m_data[i] * other.m_data[i]; 
        });

        return result;
    }

    Matrix Matrix::transpose()
    {
        Matrix result(m_cols, m_rows, 0.0);

        ThreadPool::parallelFor(0, m_rows, [&](int i) {
            for (int j = 0; j < m_cols; j++)
                result[{j, i}] = (*this)[{i, j}];
        });

        return result;
    }

    Matrix Matrix::map(std::function<double(double)> func)
    {
        Matrix result(m_rows, m_cols, 0.0);

        ThreadPool::parallelFor(0, m_rows * m_cols, [&](int i) {
            result.m_data[i] = func(m_data[i]);
        });

        return result;
    }

    Matrix &Matrix::operator=(const Matrix &other)
    {
        if (this == &other)
            return *this;
        m_rows = other.m_rows;
        m_cols = other.m_cols;
        m_data = other.m_data;
        return *this;
    }

    Matrix &Matrix::operator+=(const Matrix &other)
    {
        if (m_rows != other.m_rows || m_cols != other.m_cols)
            throw std::invalid_argument("Matrix dimensions must match for addition.");
        
        ThreadPool::parallelFor(0, m_rows * m_cols, [&](int i) {
            m_data[i] += other.m_data[i];
        });

        return *this;
    }

    Matrix &Matrix::operator+=(const double scalar)
    {
        ThreadPool::parallelFor(0, m_rows * m_cols, [&](int i) {
            m_data[i] += scalar;
        });

        return *this;
    }

    Matrix &Matrix::operator-=(const Matrix &other)
    {
        if (m_rows != other.m_rows || m_cols != other.m_cols)
            throw std::invalid_argument("Matrix dimensions must match for subtraction.");
        
        ThreadPool::parallelFor(0, m_rows * m_cols, [&](int i) {
            m_data[i] -= other.m_data[i];
        });

        return *this;
    }

    Matrix &Matrix::operator-=(const double scalar)
    {
        ThreadPool::parallelFor(0, m_rows * m_cols, [&](int i) {
            m_data[i] -= scalar;
        });

        return *this;
    }

    Matrix &Matrix::operator*=(const Matrix &other)
    {
        if (m_cols != other.m_rows)
            throw std::invalid_argument("Invalid matrix multiplication: A(m x k) * B(k x n) requires A.cols == B.rows.");

        Matrix result(m_rows, other.m_cols, 0.0);

        ThreadPool::parallelFor(0, m_rows, [&](int i) {
            for (int j = 0; j < other.m_cols; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < m_cols; k++)
                    sum += (*this)[{i, k}] * other[{k, j}];
                result[{i, j}] = sum;
            }
        });

        *this = std::move(result);
        return *this;
    }

    Matrix &Matrix::operator*=(const double scalar)
    {
        ThreadPool::parallelFor(0, m_rows * m_cols, [&](int i) {
            m_data[i] *= scalar;
        });

        return *this;
    }

    Matrix &Matrix::operator/=(const Matrix &other)
    {
        if (m_rows != other.m_rows || m_cols != other.m_cols)
            throw std::invalid_argument("Matrix dimensions must match for division.");
        
        // Check for division by zero before starting the loop
        for (const auto &val : other.m_data)
        {
            if (val == 0.0)
                throw std::runtime_error("Division by zero is not allowed.");
        }
        
        ThreadPool::parallelFor(0, m_rows * m_cols, [&](int i) {
            m_data[i] /= other.m_data[i];
        });

        return *this;
    }

    Matrix &Matrix::operator/=(const double scalar)
    {
        if (scalar == 0)
            throw std::runtime_error("Division by zero is not allowed.");
        
        ThreadPool::parallelFor(0, m_rows * m_cols, [&](int i) {
            m_data[i] /= scalar;
        });

        return *this;
    }

    std::ostream &operator<<(std::ostream &out, const Matrix &m)
    {
        const int cell_width = 10;
        const int precision = 6;

        out << (char)218 << std::string(m.getCols() * (cell_width + 1), ' ') << (char)191 << "\n";

        for (int i = 0; i < m.getRows(); i++)
        {
            out << (char)179;
            for (int j = 0; j < m.getCols(); j++)
                out << std::fixed << std::setw(cell_width) << std::setprecision(precision) << m[{i, j}] << " ";
            out << (char)179 << "\n";
        }

        out << (char)192 << std::string(m.getCols() * (cell_width + 1), ' ') << (char)217 << "\n";

        return out;
    }

    Matrix operator*(const Matrix &left, const Matrix &right)
    {
        if (left.m_cols != right.m_rows)
            throw std::invalid_argument("Invalid matrix multiplication: A(m x k) * B(k x n) requires A.cols == B.rows.");
        
        Matrix result(left.m_rows, right.m_cols, 0.0);

        ThreadPool::parallelFor(0, left.m_rows, [&](int i) {
            for (int j = 0; j < right.m_cols; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < left.m_cols; k++)
                    sum += left[{i, k}] * right[{k, j}];
                result[{i, j}] = sum;
            }
        });

        return result;
    }

    Matrix operator*(const double scalar, const Matrix &right)
    {
        Matrix result = right;
        result *= scalar;
        return result;
    }

    Matrix operator*(const Matrix &left, const double scalar)
    {
        return scalar * left;
    }

    Matrix operator+(const Matrix &left, const Matrix &right)
    {
        Matrix result = left;
        result += right;
        return result;
    }

    Matrix operator+(const double scalar, const Matrix &right)
    {
        Matrix result = right;
        result += scalar;
        return result;
    }

    Matrix operator+(const Matrix &left, const double scalar)
    {
        return scalar + left;
    }

    Matrix operator-(const Matrix &left, const Matrix &right)
    {
        Matrix result = left;
        result -= right;
        return result;
    }

    Matrix operator-(const Matrix &right, const double scalar)
    {
        Matrix result = right;
        result -= scalar;
        return result;
    }

    Matrix operator/(const Matrix &left, const Matrix &right)
    {
        Matrix result = left;
        result /= right;
        return result;
    }

    Matrix operator/(const Matrix &right, const double scalar)
    {
        Matrix result = right;
        result /= scalar;
        return result;
    }

    bool operator==(const Matrix &left, const Matrix &right)
    {
        if (left.m_rows != right.m_rows || left.m_cols != right.m_cols)
            return false;
        
        std::atomic<bool> eq = true;

        ThreadPool::parallelFor(0, left.m_rows * left.m_cols, [&](int i) {
            if (left.m_data[i] != right.m_data[i]) 
                eq.store(false, std::memory_order_relaxed);
        });

        return eq;
    }

    bool operator!=(const Matrix &left, const Matrix &right)
    {
        return !(left == right);
    }
}