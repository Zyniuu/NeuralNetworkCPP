/**
 * C++ neural network library
 *
 * Matrix.cpp
 */

#include "Matrix.hpp"
#include <thread>

namespace nn
{
    Matrix::Matrix()
        : m_rows(0), m_cols(0), m_data({}) {}
    
    Matrix::Matrix(const Matrix &matrix)
        : m_rows(matrix.m_rows), m_cols(matrix.m_cols), m_data(matrix.m_data) {}
    
    Matrix::Matrix(Matrix &&matrix)
        : m_rows(matrix.m_rows), m_cols(matrix.m_cols), m_data(std::move(matrix.m_data)) {}
    
    Matrix::Matrix(const int &rows, const int &cols, const int &initVal = 0)
        : m_rows(rows), m_cols(cols), m_data(rows * cols, initVal) {}
    
    Matrix::Matrix(const int &rows, const int &cols, const std::vector<double> &data)
        : m_rows(rows), m_cols(cols), m_data(data) {}
    
    Matrix::Matrix(const int &rows, const int &cols, std::function<double()> func)
        : m_rows(rows), m_cols(cols), m_data(rows * cols)
    {
        for (double &val : m_data)
            val = func();
    }

    double &Matrix::operator[](const std::pair<int, int> &index)
    {
        return m_data[index.first * m_cols + index.second];
    }

    const double &Matrix::operator[](const std::pair<int, int> &index) const
    {
        return m_data[index.first * m_cols + index.second];
    }

    Matrix Matrix::cwiseProduct(const Matrix &other)
    {
        Matrix result(m_rows, m_cols, 0.0);

        auto worker = [&](int start, int end)
        {
            for (int i = start; i < end; i++)
                result.m_data[i] = m_data[i] * other.m_data[i];
        };

        int numThreads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        int chunkSize = m_data.size() / numThreads;

        for (int i = 0; i < numThreads; i++)
        {
            int start = i * chunkSize;
            int end = (i == numThreads - 1) ? m_data.size() : start + chunkSize;
            threads.emplace_back(worker, start, end);
        }

        for (auto &t : threads)
            t.join();

        return result;
    }

    Matrix Matrix::transpose()
    {
        Matrix result(m_cols, m_rows, 0.0);

        for (int i = 0; i < m_rows; i++)
            for (int j = 0; j < m_cols; j++)
                result[{j, i}] = (*this)[{i, j}];

        return result;
    }

    Matrix Matrix::map(std::function<double(double)> func)
    {
        Matrix result(m_rows, m_cols, 0.0);

        auto worker = [&](int start, int end)
        {
            for (int i = start; i < end; i++)
                result.m_data[i] = func(m_data[i]);
        };

        int numThreads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        int chunkSize = m_data.size() / numThreads;

        for (int i = 0; i < numThreads; i++)
        {
            int start = i * chunkSize;
            int end = (i == numThreads - 1) ? m_data.size() : start + chunkSize;
            threads.emplace_back(worker, start, end);
        }

        for (auto &t : threads)
            t.join();

        return result;
    }
}