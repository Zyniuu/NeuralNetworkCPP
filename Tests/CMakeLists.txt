# Minimal version of CMake required
cmake_minimum_required(VERSION 3.10)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Project name
project(NeuralNetworCPPTests)

# Enable testing
enable_testing()

# Add Google Test directory
add_subdirectory(googletest)

# Include directories for headers
include_directories(
    googletest/include
    ../NeuralNetworkCPP/Initializers/Common
    ../NeuralNetworkCPP/Initializers/XavierNormal
    ../NeuralNetworkCPP/Initializers/XavierUniform
    ../NeuralNetworkCPP/Initializers/HeNormal
    ../NeuralNetworkCPP/Initializers/HeUniform
    ../NeuralNetworkCPP/GlobalThreadPool
    ../NeuralNetworkCPP/GlobalThreadPool/Base
    ../NeuralNetworkCPP/Matrix
    ../NeuralNetworkCPP/Matrix/ColumnWiseProxy
    ../NeuralNetworkCPP/Losses/Common
    ../NeuralNetworkCPP/Losses/MeanSquaredError
    ../NeuralNetworkCPP/Losses/CategoricalCrossEntropy
    ../NeuralNetworkCPP/Losses/BinaryCrossEntropy
    ../NeuralNetworkCPP/Optimizers/Common
    ../NeuralNetworkCPP/Optimizers/SGD
)

# List of source files for the tests
set(TEST_SOURCES
    Initializers/XavierNormal/TestXavierNormal.cpp
    Initializers/XavierUniform/TestXavierUniform.cpp
    Initializers/HeNormal/TestHeNormal.cpp
    Initializers/HeUniform/TestHeUniform.cpp
    ThreadPool/TestThreadPool.cpp
    Matrix/TestMatrix.cpp
    Losses/MSE/TestMSE.cpp
    Losses/CCE/TestCCE.cpp
    Losses/BCE/TestBCE.cpp
    Optimizers/TestOptimizers.cpp
    tests.cpp
)

# List of source files of the neural network
set(NN_SOURCES
    ../NeuralNetworkCPP/Initializers/XavierNormal/XavierNormal.cpp
    ../NeuralNetworkCPP/Initializers/XavierUniform/XavierUniform.cpp
    ../NeuralNetworkCPP/Initializers/HeNormal/HeNormal.cpp
    ../NeuralNetworkCPP/Initializers/HeUniform/HeUniform.cpp
    ../NeuralNetworkCPP/GlobalThreadPool/Base/ThreadPool.tpp
    ../NeuralNetworkCPP/GlobalThreadPool/Base/ThreadPool.cpp
    ../NeuralNetworkCPP/GlobalThreadPool/GlobalThreadPool.cpp
    ../NeuralNetworkCPP/Matrix/Matrix.cpp
    ../NeuralNetworkCPP/Matrix/ColumnWiseProxy/ColumnWiseProxy.cpp
    ../NeuralNetworkCPP/Losses/MeanSquaredError/MeanSquaredError.cpp
    ../NeuralNetworkCPP/Losses/CategoricalCrossEntropy/CategoricalCrossEntropy.cpp
    ../NeuralNetworkCPP/Losses/BinaryCrossEntropy/BinaryCrossEntropy.cpp
    ../NeuralNetworkCPP/Optimizers/SGD/SGD.cpp
)

# Create the test executable
add_executable(${PROJECT_NAME} ${TEST_SOURCES} ${NN_SOURCES})

# Link the test executable with Google Test and pthread
target_link_libraries(${PROJECT_NAME} gtest)

# Add test target for running tests
add_test(NAME AllTests COMMAND tests)