/**
 * C++ neural network library
 *
 * ColWiseProxy.hpp
 */

 #ifndef COLWISEPROXY_HPP
 #define COLWISEPROXY_HPP
 
 #include "../Matrix.hpp"
 
 namespace nn
 {
     // Forward declaration of Matrix class
     class Matrix;
 
     /**
      * @class ColWiseProxy
      * @brief Proxy class for performing column-wise operations on a matrix.
      *
      * This class enables operations like multiplying a column vector by all columns of a matrix.
      * It is returned by the `Matrix::colWise()` method.
      */
     class ColWiseProxy
     {
     private:
         Matrix &m_matrix; ///< Reference to the original matrix.
 
     public:
         /**
          * @brief Constructs a ColWiseProxy for the given matrix.
          *
          * @param matrix The matrix to perform column-wise operations on.
          */
         ColWiseProxy(Matrix &matrix);

        /**
         * @brief Returns a row vector with maximum coefficient of each column.
         *
         * @return The row vector with maximum coefficient of each column.
         */
         Matrix maxCoeff() const;

        /**
         * @brief Sums elements in each column.
         * 
         * @return Row matrix with summed up columns.
         */
         Matrix sum() const;
 
         /**
          * @brief Multiplies a column vector by every column of a matrix.
          *
          * @param other A column vector (Matrix with 1 column).
          * @return A new Matrix after the column-wise multiplication.
          * @throws std::invalid_argument If `other` is not a column vector or its row count
          *                               does not match the original matrix.
          */
         friend Matrix operator*(const ColWiseProxy &left, Matrix &right);

         /**
          * @brief Adds a column vector to every column of a matrix.
          *
          * @param other A column vector (Matrix with 1 column).
          * @return A new Matrix after the column-wise addition.
          * @throws std::invalid_argument If `other` is not a column vector or its row count
          *                               does not match the original matrix.
          */
         friend Matrix operator+(const ColWiseProxy &left, Matrix &right);

         /**
          * @brief Subtracts a column vector from every column of a matrix.
          *
          * @param other A column vector (Matrix with 1 column).
          * @return A new Matrix after the column-wise subtraction.
          * @throws std::invalid_argument If `other` is not a column vector or its row count
          *                               does not match the original matrix.
          */
         friend Matrix operator-(const ColWiseProxy &left, Matrix &right);
     };
 }
 
 #endif