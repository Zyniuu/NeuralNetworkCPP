/**
 * C++ neural network library
 *
 * TestMatrix.cpp
 */

#include <gtest/gtest.h>
#include <filesystem>
#include <NeuralNetworkCPP/Matrix/Matrix.hpp>
#include <NeuralNetworkCPP/Initializers/Initializers.hpp>

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

TEST(MatrixTests, ConstructFromVectorOfVectors)
{
    nn::Matrix mat({
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    });

    EXPECT_DOUBLE_EQ(mat(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(mat(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(mat(0, 2), 3.0);
    EXPECT_DOUBLE_EQ(mat(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(mat(1, 1), 5.0);
    EXPECT_DOUBLE_EQ(mat(1, 2), 6.0);
    EXPECT_DOUBLE_EQ(mat(2, 0), 7.0);
    EXPECT_DOUBLE_EQ(mat(2, 1), 8.0);
    EXPECT_DOUBLE_EQ(mat(2, 2), 9.0);
}

// Test max coeff
TEST(MatrixTests, MaxCoeff)
{
    nn::Matrix mat(3, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});

    // Test with a small matrix
    EXPECT_DOUBLE_EQ(mat.maxCoeff(), 9.0);

    // Test with a large matrix
    nn::Matrix largeMat(1000, 1000);
    for (int i = 0; i < 1000; i++)
    {
        for (int j = 0; j < 1000; j++)
        {
            largeMat[{i, j}] = i * 1000 + j;
        }
    }
    EXPECT_DOUBLE_EQ(largeMat.maxCoeff(), 999 * 1000 + 999);
}

// Test max coeff column-wise
TEST(MatrixTests, MaxCoeffColWise)
{
    nn::Matrix mat(3, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    nn::Matrix expected(1, 3, {7.0, 8.0, 9.0});

    EXPECT_EQ(mat.colWise().maxCoeff(), expected);
}

// Test matrix sum
TEST(MatrixTests, Sum)
{
    nn::Matrix mat(3, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});

    // Test with a small matrix
    EXPECT_DOUBLE_EQ(mat.sum(), 45.0);

    // Test with a large matrix
    nn::Matrix largeMat(1000, 1000, 1.0);

    EXPECT_DOUBLE_EQ(largeMat.sum(), 1000000.0);
}

// Test matrix row-wise sum
TEST(MatrixTests, RowWiseSum)
{
    nn::Matrix mat({
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    });

    nn::Matrix expectedOutput(3, 1, {6.0, 15.0, 24.0});

    EXPECT_EQ(mat.rowWise().sum(), expectedOutput);
}

// Test matrix column-wise sum
TEST(MatrixTests, ColWiseSum)
{
    nn::Matrix mat({
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    });

    nn::Matrix expectedOutput(1, 3, {12.0, 15.0, 18.0});

    EXPECT_EQ(mat.colWise().sum(), expectedOutput);
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

// Test xavier normal init
TEST(MatrixTests, XavierNormalInitialization)
{
    int rows = 1000, cols = 1000;
    int inputs = rows, outputs = cols;

    nn::XavierNormal initializer(inputs, outputs);
    nn::Matrix matrix(rows, cols, [&initializer]() { return initializer.getRandomNum(); });

    double sum = 0.0, sum_sq = 0.0;
    int total_elements = rows * cols;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            double value = matrix[{i, j}];
            sum += value;
            sum_sq += value * value;
        }
    }

    double mean = sum / total_elements;
    double variance = (sum_sq / total_elements) - (mean * mean);
    double expected_stddev = sqrt(2.0 / (inputs + outputs));

    EXPECT_NEAR(mean, 0.0, 0.05);
    EXPECT_NEAR(sqrt(variance), expected_stddev, 0.1);
}

// Test xavier uniform init
TEST(MatrixTests, XavierUniformInitialization)
{
    int rows = 1000, cols = 1000;
    int inputs = rows, outputs = cols;

    nn::XavierUniform initializer(inputs, outputs);
    nn::Matrix matrix(rows, cols, [&initializer]() { return initializer.getRandomNum(); });

    double min_value = std::numeric_limits<double>::max();
    double max_value = std::numeric_limits<double>::lowest();
    double sum = 0.0;
    int total_elements = rows * cols;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            double value = matrix[{i, j}];
            sum += value;
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
        }
    }

    double mean = sum / total_elements;
    double limit = sqrt(6.0 / (inputs + outputs));

    EXPECT_NEAR(mean, 0.0, 0.05);
    EXPECT_GE(min_value, -limit);
    EXPECT_LE(max_value, limit);
}

// Test He normal init
TEST(MatrixTests, HeNormalInitialization)
{
    int rows = 1000, cols = 1000;
    int inputs = rows, outputs = cols;

    nn::HeNormal initializer(inputs, outputs);
    nn::Matrix matrix(rows, cols, [&initializer]() { return initializer.getRandomNum(); });

    double sum = 0.0, sum_sq = 0.0;
    int total_elements = rows * cols;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            double value = matrix[{i, j}];
            sum += value;
            sum_sq += value * value;
        }
    }

    double mean = sum / total_elements;
    double variance = (sum_sq / total_elements) - (mean * mean);
    double expected_stddev = sqrt(2.0 / inputs);

    EXPECT_NEAR(mean, 0.0, 0.05);
    EXPECT_NEAR(sqrt(variance), expected_stddev, 0.1);
}

// Test He uniform init
TEST(MatrixTests, HeUniformInitialization)
{
    int rows = 1000, cols = 1000;
    int inputs = rows, outputs = cols;

    nn::HeUniform initializer(inputs, outputs);
    nn::Matrix matrix(rows, cols, [&initializer]() { return initializer.getRandomNum(); });

    double min_value = std::numeric_limits<double>::max();
    double max_value = std::numeric_limits<double>::lowest();
    double sum = 0.0;
    int total_elements = rows * cols;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            double value = matrix[{i, j}];
            sum += value;
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
        }
    }

    double mean = sum / total_elements;
    double limit = sqrt(6.0 / inputs);

    EXPECT_NEAR(mean, 0.0, 0.05);
    EXPECT_GE(min_value, -limit);
    EXPECT_LE(max_value, limit);
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

// Test saving and loading matrix from a file
TEST(MatrixTests, SaveAndLoad)
{
    nn::Matrix original(2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

    // Save the matrix to a file
    std::ofstream outFile("test_matrix.bin", std::ios::binary);
    ASSERT_TRUE(outFile.is_open());
    original.save(outFile);
    outFile.close();

    // Load the matrix from the file
    std::ifstream inFile("test_matrix.bin", std::ios::binary);
    ASSERT_TRUE(inFile.is_open());
    nn::Matrix loaded(inFile);
    inFile.close();

    // Verify that the loaded matrix matches the original
    EXPECT_EQ(original.getRows(), loaded.getRows());
    EXPECT_EQ(original.getCols(), loaded.getCols());

    for (int i = 0; i < original.getRows(); i++)
    {
        for (int j = 0; j < original.getCols(); j++)
        {
            EXPECT_DOUBLE_EQ(original(i, j), loaded(i, j));
        }
    }

    std::filesystem::remove("test_matrix.bin");
}

// Test saving to an invalid file
TEST(MatrixTests, SaveInvalidFile)
{
    nn::Matrix matrix(2, 2);
    std::ofstream outFile;

    // Attempt to save to a closed file
    EXPECT_THROW(matrix.save(outFile), std::runtime_error);
}

// Test loading from an invalid file
TEST(MatrixTests, LoadInvalidFile)
{
    std::ifstream inFile;

    // Attempt to load from a closed file
    EXPECT_THROW(nn::Matrix matrix(inFile), std::runtime_error);
}

// Test loading from a file with invalid dimensions of matrix
TEST(MatrixTests, LoadInvalidDimensions)
{
    // Create a file with invalid dimensions (negative values)
    std::ofstream outFile("invalid_dimensions.bin", std::ios::binary);
    ASSERT_TRUE(outFile.is_open());
    int rows = -1, cols = -1;
    outFile.write(reinterpret_cast<char *>(&rows), sizeof(rows));
    outFile.write(reinterpret_cast<char *>(&cols), sizeof(cols));
    outFile.close();

    // Attempt to load the file
    std::ifstream inFile("invalid_dimensions.bin", std::ios::binary);
    ASSERT_TRUE(inFile.is_open());
    EXPECT_THROW(nn::Matrix matrix(inFile), std::runtime_error);
    inFile.close();

    // Delete the file after the test
    std::filesystem::remove("invalid_dimensions.bin");
}

// Test row wise subtraction
TEST(MatrixTests, RowWiseSubtraction)
{
    // Create a 2x2 matrix: {{1, 6}, {2, 7}}
    nn::Matrix mat(2, 2, {1, 6, 2, 7});

    // Create a row vector: {{0, 1}}
    nn::Matrix rowVector(1, 2, {0, 1});

    // Perform row-wise subtraction
    nn::Matrix result = mat.rowWise() - rowVector;

    // Verify the result: {{1, 7}, {2, 8}}
    EXPECT_EQ(result.getRows(), 2);
    EXPECT_EQ(result.getCols(), 2);
    EXPECT_DOUBLE_EQ(result(0, 0), 1);
    EXPECT_DOUBLE_EQ(result(0, 1), 5);
    EXPECT_DOUBLE_EQ(result(1, 0), 2);
    EXPECT_DOUBLE_EQ(result(1, 1), 6);
}

// Test row wise division
TEST(MatrixTests, RowWiseDivision)
{
    // Create a 2x2 matrix: {{2, 4}, {6, 12}}
    nn::Matrix mat(2, 2, {2, 4, 6, 12});

    // Create a row vector: {{2, 4}}
    nn::Matrix rowVector(1, 2, {2, 4});

    // Perform row-wise division
    nn::Matrix result = mat.rowWise() / rowVector;

    // Verify the result: {{1, 1}, {3, 3}}
    EXPECT_EQ(result.getRows(), 2);
    EXPECT_EQ(result.getCols(), 2);
    EXPECT_DOUBLE_EQ(result(0, 0), 1);
    EXPECT_DOUBLE_EQ(result(0, 1), 1);
    EXPECT_DOUBLE_EQ(result(1, 0), 3);
    EXPECT_DOUBLE_EQ(result(1, 1), 3);
}

// Test row wise operations when given invalid dimensions
TEST(MatrixTests, RowWiseInvalidDimensions)
{
    nn::Matrix mat(3, 2); // 3x2 matrix
    nn::Matrix invalidVector(1, 3); // 1x3 vector (columns don't match)

    // Attempt invalid column-wise addition
    EXPECT_THROW(mat.rowWise() - invalidVector, std::invalid_argument);
}

// Test row wise operations when row vector was not provided
TEST(MatrixTests, NotARowVector)
{
    nn::Matrix mat(2, 2);
    nn::Matrix invalidMatrix(2, 3); // Not a row vector

    // Attempt invalid column-wise addition
    EXPECT_THROW(mat.rowWise() - invalidMatrix, std::invalid_argument);
}

// Test column-wise multiplication
TEST(MatrixTests, ColWiseMultiplication)
{
    // Create a 2x2 matrix: {{1, 6}, {2, 7}}
    nn::Matrix mat(2, 2, {1, 6, 2, 7});

    // Create a column vector: {{1}, {2}}
    nn::Matrix colVector(2, 1, {1, 2});

    // Perform column-wise multiplication
    nn::Matrix result = mat.colWise() * colVector;

    // Verify the result: {{1, 6}, {4, 14}}
    EXPECT_EQ(result.getRows(), 2);
    EXPECT_EQ(result.getCols(), 2);
    EXPECT_DOUBLE_EQ(result(0, 0), 1);
    EXPECT_DOUBLE_EQ(result(0, 1), 6);
    EXPECT_DOUBLE_EQ(result(1, 0), 4);
    EXPECT_DOUBLE_EQ(result(1, 1), 14);
}

// Test column-wise division
TEST(MatrixTests, ColWiseDivision)
{
    // Create a 2x2 matrix: {{1, 6}, {2, 8}}
    nn::Matrix mat(2, 2, {1, 6, 2, 8});

    // Create a column vector: {{1}, {2}}
    nn::Matrix colVector(2, 1, {1, 2});

    // Perform column-wise division
    nn::Matrix result = mat.colWise() / colVector;

    // Verify the result: {{1, 6}, {1, 4}}
    EXPECT_EQ(result.getRows(), 2);
    EXPECT_EQ(result.getCols(), 2);
    EXPECT_DOUBLE_EQ(result(0, 0), 1);
    EXPECT_DOUBLE_EQ(result(0, 1), 6);
    EXPECT_DOUBLE_EQ(result(1, 0), 1);
    EXPECT_DOUBLE_EQ(result(1, 1), 4);
}

// Test column-wise addition
TEST(MatrixTests, ColWiseAddition)
{
    // Create a 2x2 matrix: {{1, 6}, {2, 8}}
    nn::Matrix mat(2, 2, {1, 6, 2, 8});

    // Create a column vector: {{1}, {2}}
    nn::Matrix colVector(2, 1, {1, 2});

    // Perform column-wise division
    nn::Matrix result = mat.colWise() + colVector;

    // Verify the result: {{2, 7}, {4, 10}}
    EXPECT_EQ(result.getRows(), 2);
    EXPECT_EQ(result.getCols(), 2);
    EXPECT_DOUBLE_EQ(result(0, 0), 2);
    EXPECT_DOUBLE_EQ(result(0, 1), 7);
    EXPECT_DOUBLE_EQ(result(1, 0), 4);
    EXPECT_DOUBLE_EQ(result(1, 1), 10);
}

// Test column-wise subtraction
TEST(MatrixTests, ColWiseSubtraction)
{
    // Create a 2x2 matrix: {{1, 6}, {2, 8}}
    nn::Matrix mat(2, 2, {1, 6, 2, 8});

    // Create a column vector: {{1}, {2}}
    nn::Matrix colVector(2, 1, {1, 2});

    // Perform column-wise division
    nn::Matrix result = mat.colWise() - colVector;

    // Verify the result: {{0, 5}, {0, 6}}
    EXPECT_EQ(result.getRows(), 2);
    EXPECT_EQ(result.getCols(), 2);
    EXPECT_DOUBLE_EQ(result(0, 0), 0);
    EXPECT_DOUBLE_EQ(result(0, 1), 5);
    EXPECT_DOUBLE_EQ(result(1, 0), 0);
    EXPECT_DOUBLE_EQ(result(1, 1), 6);
}

// Test column-wise operations when given invalid dimensions
TEST(MatrixTests, ColWiseInvalidDimensions)
{
    nn::Matrix mat(2, 3); // 2x3 matrix
    nn::Matrix invalidVector(3, 1); // 3x1 vector (rows don't match)

    // Attempt invalid column-wise multiplication
    EXPECT_THROW(mat.colWise() * invalidVector, std::invalid_argument);
}

// Test column-wise operations when column vector was not provided
TEST(MatrixTests, NotAColVector)
{
    nn::Matrix mat(2, 2);
    nn::Matrix invalidMatrix(2, 3); // Not a column vector

    // Attempt invalid column-wise multiplication
    EXPECT_THROW(mat.colWise() * invalidMatrix, std::invalid_argument);
}

TEST(MatrixTests, IdentityMatrix)
{
    nn::Matrix id = nn::Matrix::identity(3);

    std::vector<double> expected = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

    EXPECT_EQ(expected, id.getData());
}

TEST(MatrixTests, IdentityMatrixInvalidSize)
{
    EXPECT_THROW(nn::Matrix::identity(0), std::invalid_argument);
}