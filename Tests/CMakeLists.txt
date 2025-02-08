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

# Add Neural Network directory
add_subdirectory(../NeuralNetworkCPP NeuralNetworkCPPBuild)

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
    Activations/TestActivations.cpp
    Layers/TestLayers.cpp
    tests.cpp
)

# Create the test executable
add_executable(${PROJECT_NAME} ${TEST_SOURCES})

# Link the test executable with Google Test and Neural Network
target_link_libraries(${PROJECT_NAME} gtest NeuralNetworCPP)

# Add test target for running tests
add_test(NAME AllTests COMMAND tests)
