# Minimal version of CMake required
cmake_minimum_required(VERSION 3.10)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Project name
project(BostonHousingExample)

# Set build type at Release
set(CMAKE_BUILD_TYPE Release)

# Add Neural Network directory
add_subdirectory(../../NeuralNetworkCPP NeuralNetworkCPPBuild)

# Create the xor example executable
add_executable(${PROJECT_NAME} BostonHousingExample.cpp)

# Link the executable with NeuralNetworkCPP
target_link_libraries(${PROJECT_NAME} NeuralNetworCPP)

# Setting up compilator flags for release build
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
endif()