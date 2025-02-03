/**
 * C++ neural network library
 *
 * TestMatrix.cpp
 */

#include <gtest/gtest.h>
#include "../../NeuralNetworkCPP/Matrix/Matrix.hpp"
#include "../../NeuralNetworkCPP/Initializers/XavierUniform/XavierUniform.hpp"
#include "../../NeuralNetworkCPP/Initializers/XavierNormal/XavierNormal.hpp"
#include "../../NeuralNetworkCPP/Initializers/HeNormal/HeNormal.hpp"
#include "../../NeuralNetworkCPP/Initializers/HeUniform/HeUniform.hpp"

// Test constructor with default values
TEST(MatrixTests, DefaultConstructor)
{
    nn::Matrix m;
    EXPECT_EQ(m.getRows(), 0);
    EXPECT_EQ(m.getCols(), 0);
    EXPECT_TRUE(m.getData().empty());
}

// Test constructor with given dimensions
TEST(MatrixTests, ConstructorWithSize)
{
    nn::Matrix m(3, 3, 5.0);
    EXPECT_EQ(m.getRows(), 3);
    EXPECT_EQ(m.getCols(), 3);
    for (double val : m.getData())
    {
        EXPECT_EQ(val, 5.0);
    }
}

// Test copy constructor
TEST(MatrixTests, CopyConstructor)
{
    nn::Matrix original(2, 2, 3.0);
    nn::Matrix copy(original);
    EXPECT_EQ(copy, original);
}

// Test move constructor
TEST(MatrixTests, MoveConstructor)
{
    nn::Matrix original(2, 2, 4.0);
    nn::Matrix moved(std::move(original));
    EXPECT_EQ(moved.getRows(), 2);
    EXPECT_EQ(moved.getCols(), 2);
    double val = moved[{0, 0}];
    EXPECT_EQ(moved(0, 0), 4.0);
}

// Test element-wise addition
TEST(MatrixTests, AdditionOperator)
{
    nn::Matrix A(2, 2, 2.0);
    nn::Matrix B(2, 2, 3.0);
    nn::Matrix C = A + B;

    EXPECT_EQ(C(0, 0), 5.0);
    EXPECT_EQ(C(1, 1), 5.0);
}

// Test scalar addition
TEST(MatrixTests, ScalarAddition)
{
    nn::Matrix A(2, 2, 2.0);
    nn::Matrix C = A + 3.0;

    EXPECT_EQ(C(0, 0), 5.0);
}

// Test element-wise subtraction
TEST(MatrixTests, SubtractionOperator)
{
    nn::Matrix A(2, 2, 5.0);
    nn::Matrix B(2, 2, 3.0);
    nn::Matrix C = A - B;

    EXPECT_EQ(C(0, 0), 2.0);
    EXPECT_EQ(C(1, 1), 2.0);
}

// Test scalar multiplication
TEST(MatrixTests, ScalarMultiplication)
{
    nn::Matrix A(2, 2, 2.0);
    nn::Matrix C = 3.0 * A;

    EXPECT_EQ(C(0, 0), 6.0);
    EXPECT_EQ(C(1, 1), 6.0);
}

// Test matrix multiplication
TEST(MatrixTests, MatrixMultiplication)
{
    nn::Matrix A(2, 3, {1, 2, 3, 4, 5, 6});
    nn::Matrix B(3, 2, {7, 8, 9, 10, 11, 12});
    nn::Matrix C = A * B;

    EXPECT_EQ(C(0, 0), 58);  // 1*7 + 2*9 + 3*11
    EXPECT_EQ(C(1, 1), 154); // 4*8 + 5*10 + 6*12
}

// Test element-wise multiplication (Hadamard product)
TEST(MatrixTests, ElementWiseMultiplication)
{
    nn::Matrix A(2, 2, {1, 2, 3, 4});
    nn::Matrix B(2, 2, {2, 2, 2, 2});
    nn::Matrix C = A.cwiseProduct(B);

    EXPECT_EQ(C(0, 0), 2);
    EXPECT_EQ(C(1, 1), 8);
}

// Test transpose function
TEST(MatrixTests, Transpose)
{
    nn::Matrix A(2, 3, {1, 2, 3, 4, 5, 6});
    nn::Matrix T = A.transpose();

    EXPECT_EQ(T.getRows(), 3);
    EXPECT_EQ(T.getCols(), 2);
    EXPECT_EQ(T(0, 1), 4); // Transposed value at (0,1) should be (1,0)
}

// Test division by scalar
TEST(MatrixTests, ScalarDivision)
{
    nn::Matrix A(2, 2, 6.0);
    nn::Matrix C = A / 2.0;

    EXPECT_EQ(C(0, 0), 3.0);
}

// Test division by zero exception
TEST(MatrixTests, DivisionByZero)
{
    nn::Matrix A(2, 2, 6.0);

    EXPECT_THROW(A / 0.0, std::runtime_error);
}

// Test element access operator
TEST(MatrixTests, ElementAccess)
{
    nn::Matrix A(2, 2, {1, 2, 3, 4});

    double val1 = A[{0, 0}];
    double val2 = A[{1, 1}];
    EXPECT_EQ(val1, 1);
    EXPECT_EQ(val2, 4);
}

// Test invalid matrix multiplication dimensions
TEST(MatrixTests, InvalidMultiplication)
{
    nn::Matrix A(2, 3);
    nn::Matrix B(4, 2);

    EXPECT_THROW(A * B, std::invalid_argument);
}

// Test equality operator
TEST(MatrixTests, EqualityOperator)
{
    nn::Matrix A(2, 2, {1, 2, 3, 4});
    nn::Matrix B(2, 2, {1, 2, 3, 4});

    EXPECT_TRUE(A == B);
}

// Test inequality operator
TEST(MatrixTests, InequalityOperator)
{
    nn::Matrix A(2, 2, {1, 2, 3, 4});
    nn::Matrix B(2, 2, {1, 2, 3, 5});

    EXPECT_TRUE(A != B);
}

// Test xavier uniform init
TEST(MatrixTests, XavierUniformInitialization)
{
    int rows = 5;
    int cols = 5;
    nn::XavierUniform initializer(rows, cols);
    nn::Matrix matrix(rows, cols, [&initializer]() { 
        return initializer.getRandomNum(); 
    });

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            double value = matrix(i, j);
            EXPECT_GE(value, -1.0);
            EXPECT_LE(value, 1.0);
        }
    }
}

// Test xavier normal init
TEST(MatrixTests, XavierNormalInitialization) {
    int rows = 5, cols = 5;
    nn::XavierNormal initializer(rows, cols);
    
    nn::Matrix m(rows, cols, [&initializer]() { 
        return initializer.getRandomNum(); 
    });
    
    EXPECT_EQ(m.getRows(), rows);
    EXPECT_EQ(m.getCols(), cols);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double value = m(i, j);
            EXPECT_GE(value, -1.0);
            EXPECT_LE(value, 1.0);
        }
    }
}

// Test He normal init
TEST(MatrixTests, HeNormalInitialization)
{
    int rows = 5;
    int cols = 5;
    nn::HeNormal initializer(rows, cols);
    nn::Matrix matrix(rows, cols, [&initializer]() { 
        return initializer.getRandomNum(); 
    });

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            double value = matrix(i, j);
            EXPECT_NEAR(value, 0.0, 2.0);
        }
    }
}

// Test He uniform init
TEST(MatrixTests, HeUniformInitialization) {
    int rows = 5, cols = 5;
    nn::HeUniform initializer(rows, cols);
    
    nn::Matrix m(rows, cols, [&initializer]() { 
        return initializer.getRandomNum(); 
    });
    
    EXPECT_EQ(m.getRows(), rows);
    EXPECT_EQ(m.getCols(), cols);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double value = m(i, j);
            EXPECT_GE(value, -2.0);
            EXPECT_LE(value, 2.0);
        }
    }
}

// Test function mapping (e.g., squaring elements)
TEST(MatrixTests, MapFunction)
{
    nn::Matrix A(2, 2, {1, 2, 3, 4});
    nn::Matrix C = A.map([](double x) { 
        return x * x; 
    });

    EXPECT_EQ(C(0, 0), 1);
    EXPECT_EQ(C(1, 1), 16);
}

// Test large matrix creation and basic access
TEST(MatrixTests, LargeMatrixCreation)
{
    const int size = 1000;
    nn::Matrix A(size, size, 1.0);

    EXPECT_EQ(A.getRows(), size);
    EXPECT_EQ(A.getCols(), size);
    EXPECT_EQ(A(500, 500), 1.0);
}

// Test large matrix addition performance
TEST(MatrixTests, LargeMatrixAddition)
{
    const int size = 1000;
    nn::Matrix A(size, size, 2.0);
    nn::Matrix B(size, size, 3.0);
    nn::Matrix C = A + B;

    EXPECT_EQ(C(500, 500), 5.0);
}

// Test large matrix multiplication performance
TEST(MatrixTests, LargeMatrixMultiplication)
{
    const int size = 500;
    nn::Matrix A(size, size, 1.0);
    nn::Matrix B(size, size, 1.0);
    nn::Matrix C = A * B;

    EXPECT_EQ(C(0, 0), size);
}

// Test transposition performance on large matrix
TEST(MatrixTests, LargeMatrixTranspose)
{
    const int size = 1000;
    nn::Matrix A(size, size);

    // Fill matrix with increasing numbers
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            A[{i, j}] = i * size + j;

    nn::Matrix T = A.transpose();

    EXPECT_EQ(T(0, 1), A(1, 0));
    EXPECT_EQ(T(999, 500), A(500, 999));
}

// Test numerical stability (small floating-point values)
TEST(MatrixTests, NumericalStabilitySmallValues)
{
    nn::Matrix A(2, 2, {1e-9, 2e-9, 3e-9, 4e-9});
    nn::Matrix B(2, 2, {1e-9, -2e-9, 3e-9, -4e-9});
    nn::Matrix C = A + B;

    EXPECT_NEAR(C(0, 0), 2e-9, 1e-12);
    EXPECT_NEAR(C(1, 1), 0.0, 1e-12);
}