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

namespace nn
{
    class Matrix
    {
    private:
        int m_rows;
        int m_cols;
        std::vector<double> m_data;

    public:
        Matrix();
        Matrix(const Matrix &matrix);
        Matrix(Matrix &&matrix);
        Matrix(const int &rows, const int &cols, const int &initVal = 0);
        Matrix(const int &rows, const int &cols, const std::vector<double> &data);
        Matrix(const int &rows, const int &cols, std::function<double()> func);
        ~Matrix() = default;

        int getRows() const { return m_rows; }
        int getCols() const { return m_cols; }
        std::vector<double> getData() const { return m_data; }

        Matrix cwiseProduct(const Matrix &other);
        Matrix transpose();
        Matrix map(std::function<double(double)> func);

        Matrix &operator=(const Matrix &other);

        double &operator[](const std::pair<int, int> &index);
        const double &operator[](const std::pair<int, int> &index) const;

        Matrix &operator+=(const Matrix &other);
        Matrix &operator+=(const double &scalar);
        Matrix &operator-=(const Matrix &other);
        Matrix &operator-=(const double &scalar);
        Matrix &operator*=(const Matrix &other);
        Matrix &operator*=(const double &scalar);
        Matrix &operator/=(const double &scalar);

        friend std::ostream &operator<<(std::ostream &out, const Matrix &m);

        friend Matrix operator*(const Matrix &left, const Matrix &right);
        friend Matrix operator*(const double &scalar, const Matrix &right);
        friend Matrix operator*(const Matrix &left, const double &scalar);
        friend Matrix operator+(const Matrix &left, const Matrix &right);
        friend Matrix operator+(const double &scalar, const Matrix &right);
        friend Matrix operator+(const Matrix &left, const double &scalar);
        friend Matrix operator-(const Matrix &left, const Matrix &right);
        friend Matrix operator-(const double &scalar, const Matrix &right);
        friend Matrix operator-(const Matrix &left, const double &scalar);
        friend Matrix operator/(const Matrix &left, const Matrix &right);
        friend Matrix operator/(const double &scalar, const Matrix &right);
        friend Matrix operator/(const Matrix &left, const double &scalar);

        friend bool operator==(const Matrix &left, const Matrix &right);
        friend bool operator!=(const Matrix &left, const Matrix &right);
    };
}

#endif