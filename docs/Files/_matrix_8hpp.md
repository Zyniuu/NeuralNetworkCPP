# Matrix/Matrix.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::Matrix](../Classes/classnn_1_1_matrix.md)** <br>Represents a mathematical matrix with element-wise operations.  |




## Source code

```cpp


#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <functional>
#include <iostream>
#include <fstream>
#include "RowWiseProxy/RowWiseProxy.hpp"
#include "ColWiseProxy/ColWiseProxy.hpp"

namespace nn
{
    // Forward declaration of ColWiseProxy and RowWiseProxy
    class ColWiseProxy;
    class RowWiseProxy;

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

        Matrix(const int rows, const int cols, const double initVal = 0.0);

        Matrix(const int rows, const int cols, const std::vector<double> &data);

        Matrix(const std::vector<std::vector<double>> &data);

        Matrix(const int rows, const int cols, std::function<double()> func);

        Matrix(std::ifstream &file);

        ~Matrix() = default;

        int getRows() const { return m_rows; }

        int getCols() const { return m_cols; }

        std::vector<double> getData() const { return m_data; }

        void save(std::ofstream &file) const;

        double maxCoeff() const;

        double sum() const;

        RowWiseProxy rowWise();

        RowWiseProxy rowWise() const;

        ColWiseProxy colWise();

        ColWiseProxy colWise() const;

        Matrix cwiseProduct(const Matrix &other) const;

        Matrix transpose();

        Matrix map(std::function<double(double)> func) const;

        static Matrix identity(int size);

        Matrix &operator=(const Matrix &other);

        double &operator[](const std::pair<int, int> &index);
        const double &operator[](const std::pair<int, int> &index) const;

        double &operator()(const int row, const int col);
        const double &operator()(const int row, const int col) const;

        Matrix &operator+=(const Matrix &other);
        Matrix &operator+=(const double scalar);
        Matrix &operator-=(const Matrix &other);
        Matrix &operator-=(const double scalar);
        Matrix &operator*=(const Matrix &other);
        Matrix &operator*=(const double scalar);
        Matrix &operator/=(const Matrix &other);
        Matrix &operator/=(const double scalar);

        friend std::ostream &operator<<(std::ostream &out, const Matrix &m);

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

        friend bool operator==(const Matrix &left, const Matrix &right);
        friend bool operator!=(const Matrix &left, const Matrix &right);
    };
}

#endif
```
