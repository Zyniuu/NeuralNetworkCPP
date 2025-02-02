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
)

# List of source files for the tests
set(TEST_SOURCES
    Initializers/XavierNormal/TestXavierNormal.cpp
    tests.cpp
)

# List of source files of the neural network
set(NN_SOURCES
    ../NeuralNetworkCPP/Initializers/XavierNormal/XavierNormal.cpp
)

# Create the test executable
add_executable(${PROJECT_NAME} ${TEST_SOURCES} ${NN_SOURCES})

# Link the test executable with Google Test and pthread
target_link_libraries(${PROJECT_NAME} gtest)

# Add test target for running tests
add_test(NAME AllTests COMMAND tests)