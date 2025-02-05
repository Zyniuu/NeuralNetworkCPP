# Minimal version of CMake required
cmake_minimum_required(VERSION 3.10)

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
)

# List of source files for the tests
set(TEST_SOURCES
    Initializers/XavierNormal/TestXavierNormal.cpp
    Initializers/XavierUniform/TestXavierUniform.cpp
    Initializers/HeNormal/TestHeNormal.cpp
    Initializers/HeUniform/TestHeUniform.cpp
    ThreadPool/TestThreadPool.cpp
    Matrix/TestMatrix.cpp
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
)

# Create the test executable
add_executable(${PROJECT_NAME} ${TEST_SOURCES} ${NN_SOURCES})

# Link the test executable with Google Test and pthread
target_link_libraries(${PROJECT_NAME} gtest)

# Add test target for running tests
add_test(NAME AllTests COMMAND tests)